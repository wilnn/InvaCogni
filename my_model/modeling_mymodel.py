import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, RobertaModel, Wav2Vec2Model
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from dataclasses import dataclass
from typing import Optional, Tuple


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grl_lambda):
        ctx.grl_lambda = grl_lambda
        #return x.view_as(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.grl_lambda)*grad_output.neg(), None
    

class GRL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grl_lambda = config.grl_lambda
    
    def forward(self, x):
        # Use the custom autograd function
        return GradReverse.apply(x, self.grl_lambda)
    
class MyModelVisionEncoder(nn.Module):
    def __init__(self, config, vision_encoder=None):
        super().__init__()
        if vision_encoder:
            self.vision_encoder = vision_encoder
        else:
            config_ = AutoConfig.from_pretrained(config.vision_encoder_path)
            self.vision_encoder = AutoModel.from_config(config_).vision_model
    
    def forward(self, pixel_values):
        return self.vision_encoder(pixel_values)

# TODO: CHANGE TO BERT IF SLIPLP TEXT MODEL TOKE LIMIT IS TOO SMALL (ONLY 77)
class MyModelTextEncoder(nn.Module):
    def __init__(self, config, text_encoder=None):
        super().__init__()
        if text_encoder:
            self.text_encoder = text_encoder
        else:
            config_ = AutoConfig.from_pretrained(config.text_encoder_path)
            #self.text_encoder = AutoModel.from_config(config_).text_model # FOR SIGLIP TEXT ENCODER
            
            self.text_encoder = RobertaModel.from_config(config_) # FOR BERT
                                                            # can not use AutoModel here because the `architectures`
                                                            # in the config specify BertModelForCTC instead of BertModel class
                                                            # so if use AutoModel it will load BertModelForCTC model class 
    def forward(self, input_ids):
        return self.text_encoder(input_ids)
    
class MyModelAudioEncoder(nn.Module):
    def __init__(self, config, audio_encoder=None):
        super().__init__()
        if audio_encoder:
            self.audio_encoder = audio_encoder
        else:
            config_ = AutoConfig.from_pretrained(config.audio_encoder_path)
            self.audio_encoder = Wav2Vec2Model.from_config(config_)
    
    def forward(self, audio, return_dict=False):
        return self.audio_encoder(audio, return_dict=return_dict)


class MyModelFFN(nn.Module):
    def __init__(self, config, dims_per_layer: list[list]):
        super().__init__()

        act_list = {"relu":nn.ReLU(),
                    "gelu":nn.GELU(),
                    "silu":nn.SiLU(),
                    } # JUST USE GELU TO MATCH WITH THE PRETRAINED MODEL

        self.FFN = nn.Sequential(*[layer
                            for in_features, out_features in dims_per_layer
                            for layer in (nn.Linear(in_features, out_features), act_list[config.hidden_act])
                            ])
    def forward(self, x):
        return self.FFN(x)

'''
# more custom version of the cross attention
class MyModelCrossAttention2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(in_features=config.hidden_size,
                                out_features=config.hidden_size)
        
        self.k_proj = nn.Linear(in_features=config.hidden_size,
                                out_features=config.hidden_size)
        
        self.v_proj = nn.Linear(in_features=config.hidden_size,
                                out_features=config.hidden_size)
        self.out_proj = nn.Linear(in_features=config.hidden_size,
                                out_features=config.hidden_size)
        
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)# use 0.1?
        self.num_attention_heads = config.num_attention_heads

        if config.hidden_size % config.num_attention_heads == 0:
            self.head_dim = config.hidden_size // self.num_attention_heads
        else:
            raise ValueError(f"Hidden size must be divisible by num heads. Got hidden_size: {config.hidden_size} and num_attention_heads: {config.num_attention_heads}")
        
        #self.MHA = nn.MultiheadAttention(embed_dim=embed_dim,
                                        # num_heads=num_attention_heads,
                                        # batch_first=True)


    def forward(self, pixel_values_embed, input_ids_embed,
                key_attention_mask=None,
                query_attention_mask=None):
        q = self.q_proj(input_ids_embed)
        k = self.k_proj(pixel_values_embed)
        v = self.v_proj(pixel_values_embed)

        batch_size, input_ids_seq_length, embed_dim = input_ids_embed.shape
        pixel_values_seq_length = pixel_values_embed.shape[1]

        q = q.view(batch_size, input_ids_seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, pixel_values_seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, pixel_values_seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        if key_attention_mask is not None:
            # bool mask where True=keep
            key_attention_mask = key_attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (B,1,1,L_k)
            key_attention_mask = key_attention_mask.expand(batch_size, self.num_attention_heads, input_ids_seq_length, pixel_values_seq_length)                  # broadcast to (B,H,L_q,L_k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=key_attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout, 
            #is_causal=is_causal
        )
        
        #if query_attention_mask is not None:
            # query_attention_mask is of shape (B, L_q) with True = keep, False = pad
            #attn_out = attn_out * query_attention_mask.unsqueeze(1).unsqueeze(-1)
        
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, input_ids_seq_length, self.num_attention_heads*self.head_dim).contiguous()
        #return attn_out
        return self.out_proj(attn_out)
'''

class MyModelSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # batch_first=True makes input shape (B, L, E)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )

    def forward(self, input_ids_embed,
                key_attention_mask=None,  # shape (B, L_k), True=pad
                need_weights=False):

        # nn.MultiheadAttention expects float masks, True=pad
        # It automatically handles Q/K/V projection
        attn_output, attn_weights = self.mha(
            query=input_ids_embed,      # (B, L_q, E)
            key=input_ids_embed,     # (B, L_k, E)
            value=input_ids_embed,   # (B, L_k, E)
            key_padding_mask=key_attention_mask,  # True=pad positions
            need_weights=need_weights
        )

        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output

class MyModelCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # batch_first=True makes input shape (B, L, E)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )

    def forward(self, pixel_values_embed, input_ids_embed,
                key_attention_mask=None,  # shape (B, L_k), True=pad
                need_weights=False):

        # nn.MultiheadAttention expects float masks, True=pad
        # It automatically handles Q/K/V projection
        attn_output, attn_weights = self.mha(
            query=input_ids_embed,      # (B, L_q, E)
            key=pixel_values_embed,     # (B, L_k, E)
            value=pixel_values_embed,   # (B, L_k, E)
            key_padding_mask=key_attention_mask,  # True=pad positions
            need_weights=need_weights
        )

        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output

@dataclass
class MyModelSequenceClassifierOutput(SequenceClassifierOutput):
    # use Optional[...] so it's backward-compatible when field is not present
    dc_logits: Optional[torch.Tensor] = None

class MyModel(nn.Module):
    def __init__(self, config, vision_encoder=None, text_encoder=None,
                 audio_encoder=None):
        super().__init__()
        self.config = config # if use PretrainedMoDel as parent class then
                            # don't have to do this

        self.GRL = GRL(config)
        self.vision_encoder = MyModelVisionEncoder(config, vision_encoder)
        self.text_encoder = MyModelTextEncoder(config, text_encoder)
        self.audio_encoder = MyModelAudioEncoder(config, audio_encoder)


        self.audio_FFN = MyModelFFN(config, config.audio_FFN)

        self.domain_classifier = MyModelFFN(config, config.domain_classifier_FFN)
        self.task_classifier = MyModelFFN(config, config.task_classifier_FFN)
        
        self.cross_attn = MyModelCrossAttention(config)
        self.cross_attn_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)
        self.cross_attn_FFN = MyModelFFN(config, config.cross_attn_FFN)
        self.cross_attn_FFN_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)
        #self.self_attn = MyModelSelfAttention(config)
        #self.self_attn_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)


    def domain_classify(self, audio_pooled_embed):
        return self.domain_classifier(self.GRL(audio_pooled_embed))

    def task_classify(self, input_ids_out, pixel_values_out,
                      audio_pooled_embed,
                      pixel_values_attention_mask,
                      input_ids_attention_mask=None):

        '''out = self.cross_attn(pixel_values_out.last_hidden_state,
                              input_ids_out.last_hidden_state,
                              pixel_values_attention_mask)'''
        
        audio_pooled_embed = self.audio_FFN(audio_pooled_embed)

        out = self.cross_attn(pixel_values_out.last_hidden_state,
                              input_ids_out.pooler_output, # only do it with the pooler ouput since 
                                                        # we will only use it afterward
                              pixel_values_attention_mask)
        
        # residual connection and layer norm
        out = self.cross_attn_layer_norm(out + input_ids_out.last_hidden_state)

        #out = self.self_attn(out, input_ids_attention_mask)
        #out = self.self_attn_layer_norm(out + input_ids_out.last_hidden_state)

        out2 = self.cross_attn_FFN(out)
        # residual connection and layer norm
        out2 = self.cross_attn_FFN_layer_norm(out+out2)

        # concat the pooled text embedding with the pooled audio embedding
        out2 = torch.cat([out2, audio_pooled_embed], dim=-1)

        return self.task_classifier(out2)

    def forward(self, input_ids, pixel_values, audio,
                #pixel_values_attention_mask=None,
                input_ids_attention_mask=None,
                dc_labels =None,
                tc_labels =None,
                return_dict=True,
                  **kwargs):
        
        # TODO: DONE
            # WORD2VEC2 AND WHISPER DO NOT HAVE POOLED OUTPUT.
            # PEOPLE COMPUTE THE POOLED OUTPUT FROM IT BY COMPUTE THE MEAN OF
            # THE EMBEDDINGS
            # >>> pooled_output = hidden_states.mean(dim=1)
            # >>> logits = self.classifier(pooled_output)

        # TODO: DONE
            # must do return_dict = True so that the word2vec model will 
            # return dict instead of tuple (only for this model. siglip do not
            # have this problem)

        audio_out = self.audio_encoder(audio, return_dict=True).last_hidden_states
        audio_hidden_states = audio_out.last_hidden_state

        audio_pooled_embed = audio_hidden_states.mean(dim=1)

        # feature extracted by convolutional in the model. Contain raw acounstic features
        #audio_feature = audio_out.extract_features

        input_ids_out = self.text_encoder(input_ids)

        pixel_values_out = self.vision_encoder(pixel_values)
        pixel_values_attention_mask = torch.ones((pixel_values_out.last_hidden_state.shape[0],
                                                  pixel_values_out.last_hidden_state.shape[1]),
                                                  dtype=torch.bool, device=pixel_values_out.device)
        
        if not self.training: # if not training
            dc_logits = self.domain_classify(audio_pooled_embed)

        tc_logits = self.task_classify(input_ids_out,
                                       pixel_values_out,
                                       audio_pooled_embed,
                                       pixel_values_attention_mask,
                                       input_ids_attention_mask)

        

        # TODO: DONE
            # residual connection and layer norm(RMSnorm?) after attention
        
        # TODO: DONE
        # only the text pooled emebddings will do the attention?  YES

        # TODO: DONE
            # use image attention mask(like when image have padded patches)
            # as key attention mask
            # use input_ids attention mask as query attention mask
        # TODO: DONE
            # can cross attention between text and image be multi block? SHOULD NOT(BLIP MODEL DON'T DO MULTIPLE CROSS ATTENTION BLOCK. ONLY 1 TO CROSS ATTEND IMAGE AND TEXT)
            # like in multi block attention, normally you use the output from the
            # previous block to the new block, but, how can cross attention
            # do multiblock when only the text output is outputed not the image?
        
        # TODO: IMPORTANT
            # IF USE THE torch.nn.MultiheadAttention then need to flip the 
            # attention mask output by the model processor because hugginface 
            # 1 and 0 in the attention mask means the opposite



        dc_loss = 0
        total_loss = 0
        tc_loss = 0

        if dc_labels:
            #loss_fct = CrossEntropyLoss()
            dc_loss = F.cross_entropy(dc_logits, dc_labels) # may have more than 2 domains/groups

        if tc_labels:
            tc_loss = F.binary_cross_entropy(F.sigmoid(tc_logits), tc_labels)

        if dc_labels or tc_labels:
            total_loss = tc_loss + self.config.loss_lambda*dc_loss

        if not return_dict:
            output = (tc_logits, dc_logits)
            return ((total_loss,) + output) if total_loss is not None else output

        return MyModelSequenceClassifierOutput(
            loss=total_loss,
            logits=tc_logits,
            dc_logits=dc_logits,
            #hidden_states=None,
            #attentions=None,
        )
