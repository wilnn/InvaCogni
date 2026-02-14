import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, RobertaModel, Wav2Vec2Model
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from dataclasses import dataclass
from typing import Optional, Tuple
from huggingface_hub import PyTorchModelHubMixin

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grl_lambda):
        ctx.grl_lambda = grl_lambda
        #return x.view_as(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.grl_lambda)*grad_output, None
    

class InvaCogniGRL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grl_lambda = config.grl_lambda
    
    def forward(self, x):
        # Use the custom autograd function
        return GradReverse.apply(x, self.grl_lambda)
    
class InvaCogniVisionEncoder(nn.Module):
    def __init__(self, config, vision_encoder=None):
        super().__init__()
        if vision_encoder:
            self.vision_encoder = vision_encoder
        else:
            config_ = AutoConfig.from_pretrained(config.vision_encoder_path)
            self.vision_encoder = AutoModel.from_config(config_).vision_model
    
    def forward(self, pixel_values, **kwargs):
        return self.vision_encoder(pixel_values, **kwargs)

class InvaCogniTextEncoder(nn.Module):
    def __init__(self, config, text_encoder=None):
        super().__init__()
        if text_encoder:
            #print(type(text_encoder))
            #exit(0)
            self.text_encoder = text_encoder
        else:
            config_ = AutoConfig.from_pretrained(config.text_encoder_path)
            #self.text_encoder = AutoModel.from_config(config_).text_model # FOR SIGLIP TEXT ENCODER
            
            self.text_encoder = AutoModel.from_config(config_)
            
    def forward(self, input_ids, attention_mask, **kwargs):
        return self.text_encoder(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 **kwargs,)
    
class InvacogniAudioEncoder(nn.Module):
    def __init__(self, config, audio_encoder=None):
        super().__init__()
        if audio_encoder:
            self.audio_encoder = audio_encoder
        else:
            config_ = AutoConfig.from_pretrained(config.audio_encoder_path)
            self.audio_encoder = AutoModel.from_config(config_).encoder
    
    def forward(self, audio, return_dict=False, **kwargs):
        return self.audio_encoder(audio,
                                  return_dict=return_dict, **kwargs)


class InvaCogniFFN(nn.Module):
    def __init__(self, config, dims_per_layer: list[list], last_act=True):
        super().__init__()

        _map = {"relu": nn.ReLU,
                "gelu": nn.GELU,
                "silu": nn.SiLU,
                "dropout": nn.Dropout,
                } # JUST USE GELU TO MATCH WITH THE PRETRAINED MODEL

        ls = []

        for ll in dims_per_layer:
            if type(ll) is not str:
                #print(type(ll[0]), type(ll[1]))
                #exit(0)
                ls.append(nn.Linear(ll[0], ll[1]))
            else:
                if ll.startswith("dropout-"):
                    temp = ll.split("-")
                    ls.append(_map[temp[0]](float(temp[1])))
                else:
                    ls.append(_map[ll]())

        self.FFN = nn.Sequential(*ls)
    def forward(self, x):
        return self.FFN(x)


# more custom version of the cross attention
class InvaCogniAttention2(nn.Module):
    def __init__(self, attention_dropout=0.0, hidden_size=None,
                 num_attention_heads=1, qdim=None, kdim=None,
                 vdim=None, projdim=None, out_features_projvdim=None,
                 out_features_projoutdim=None):
        super().__init__()
        '''
        hidden_size will replace any of these parameters if they are None: qdim, 
        vdim, projdim, out_features_projvdim, out_features_projoutdim
        hidden_size can be used when all k, q, v inputs are of the same shape and
        you want everything else to be of the same hidden_size shape even the output
        '''
        temp1 = projdim if projdim is not None else hidden_size
        temp2 = out_features_projvdim if out_features_projvdim is not None else hidden_size
        if (temp1 % num_attention_heads == 0 and
            temp2 % num_attention_heads == 0):
            
            self.head_dim = temp1 // num_attention_heads
            self.v_head_dim = temp2 // num_attention_heads
        else:
            raise ValueError(f"projdim and out_features_projvdim (or only use hidden size) must be divisible by num heads. Got projdim: {projdim}, out_features_vdim: {out_features_projvdim}, hidden_size: {hidden_size} and num_attention_heads: {num_attention_heads}")
        
        self.q_proj = nn.Linear(in_features=qdim if qdim is not None else hidden_size,
                                out_features=projdim if projdim is not None else hidden_size)
        
        self.k_proj = nn.Linear(in_features=kdim if kdim is not None else hidden_size,
                                out_features=projdim if projdim is not None else hidden_size)
        
        self.v_proj = nn.Linear(in_features=vdim if vdim is not None else hidden_size,
                                out_features=out_features_projvdim if out_features_projvdim is not None else hidden_size)
        self.out_proj = nn.Linear(in_features=out_features_projvdim if out_features_projvdim is not None else hidden_size,
                                out_features=out_features_projoutdim if out_features_projoutdim is not None else hidden_size)
        
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads

        #self.MHA = nn.MultiheadAttention(embed_dim=embed_dim,
                                        # num_heads=num_attention_heads,
                                        # batch_first=True)


    def forward(self, k, v, q,
                key_attention_mask=None,
                #query_attention_mask=None,
                is_causal=False,
                ):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        batch_size_q, seq_q, embed_dim_q = q.shape
        _, seq_k, embed_dim_k = k.shape
        _, seq_v, embed_dim_v = v.shape


        q = q.view(batch_size_q, seq_q, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size_q, seq_k, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size_q, seq_v, self.num_attention_heads, self.v_head_dim).transpose(1, 2)

        if key_attention_mask is not None:
            # bool mask where True=keep
            key_attention_mask = key_attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (B,1,1,L_k)
            key_attention_mask = key_attention_mask.expand(batch_size_q,
                                                        self.num_attention_heads,
                                                        seq_q, seq_k)  # broadcast to (B,H,L_q,L_k)
        if is_causal:
            causal_mask = torch.tril(torch.ones((seq_q, seq_k),
                                                dtype=torch.bool,
                                                device=k.device)) # shape: (L_q, L_k)

            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L_q,L_k)
            causal_mask = causal_mask.expand(batch_size_q, self.num_attention_heads, seq_q, seq_k)  # (B,H,L_q,L_k)
            if key_attention_mask is not None:
                key_attention_mask = key_attention_mask & causal_mask  # True where allowed
            else:
                key_attention_mask = causal_mask

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=key_attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout, 
            #is_causal=is_causal
        )
        
        #if query_attention_mask is not None:
            # query_attention_mask is of shape (B, L_q) with True = keep, False = pad
            #attn_out = attn_out * query_attention_mask.unsqueeze(1).unsqueeze(-1)
        
        attn_out = attn_out.transpose(1, 2).reshape(batch_size_q, seq_q, self.num_attention_heads*self.v_head_dim).contiguous()
        #return attn_out
        return self.out_proj(attn_out)


class InvaCogniCrossAttention(nn.Module):
    def __init__(self, config, kdim=768, vdim=768):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.kdim = kdim
        self.vdim = vdim
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # batch_first=True makes input shape (B, L, E)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            kdim=kdim,
            vdim=vdim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )

    def forward(self, k, q,
                key_attention_mask=None,  # shape (B, L_k), True=pad
                need_weights=False):

        # nn.MultiheadAttention expects bool or float masks, True=pad
        # It automatically handles Q/K/V projection
        attn_output, attn_weights = self.mha(
            query=q,      # (B, L_q, E)
            key=k,     # (B, L_k, E)
            value=k,   # (B, L_k, E)
            key_padding_mask=key_attention_mask,  # True=pad positions
            need_weights=need_weights
        )

        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output

@dataclass
class InvaCogniClassifierOutput(SequenceClassifierOutput):
    # use Optional[...] so it's backward-compatible when field is not present
    gender_dc_logits: Optional[torch.Tensor] = None
    gender_dc_loss: Optional[torch.Tensor] = None
    tc_loss: Optional[torch.Tensor] = None
    language_dc_logits: Optional[torch.Tensor] = None
    language_dc_loss: Optional[torch.Tensor] = None

#class InvaCogni(nn.Module, PyTorchModelHubMixin):
class InvaCogni(nn.Module):
    def __init__(self, config, vision_encoder=None, text_encoder=None,
                 audio_encoder=None):
        super().__init__()
        self.config = config # if use PretrainedMoDel as parent class then
                            # don't have to do this

        self.gender_GRL = InvaCogniGRL(config)
        self.language_GRL = InvaCogniGRL(config)
        self.vision_encoder = InvaCogniVisionEncoder(config, vision_encoder)
        self.text_encoder = InvaCogniTextEncoder(config, text_encoder)
        self.audio_encoder = InvacogniAudioEncoder(config, audio_encoder)

        self.audio_FFN = InvaCogniFFN(config, config.audio_FFN)

        self.gender_domain_classifier = InvaCogniFFN(config, config.gender_domain_classifier_FFN)
        self.language_domain_classifier = InvaCogniFFN(config, config.language_domain_classifier_FFN)
        self.task_classifier = InvaCogniFFN(config, config.task_classifier_FFN)
        
        self.cross_attn = InvaCogniCrossAttention(config)
        self.cross_attn_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)
        self.cross_attn_FFN = InvaCogniFFN(config, config.cross_attn_FFN)
        self.cross_attn_FFN_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)
        #self.self_attn = MyModelSelfAttention(config)
        #self.self_attn_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)


    def gender_domain_classify(self, audio_pooled_embed):
        return self.gender_domain_classifier(self.gender_GRL(audio_pooled_embed))
    
    def language_domain_classify(self, input_ids_out, audio_pooled_embed, input_ids_attention_mask):
        #print(input_ids_out.pooler_output.shape)
        #print(audio_pooled_embed.shape)
        input_ids_attention_mask = input_ids_attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        input_ids_out = input_ids_out.last_hidden_state * input_ids_attention_mask
        input_ids_out = input_ids_out.sum(dim=1)
        lengths = input_ids_attention_mask.sum(dim=1)
        input_ids_out = input_ids_out / lengths.clamp(min=1e-6)
        
        #input_ids_out = input_ids_out.last_hidden_state.mean(dim=1)
        out = torch.cat([input_ids_out, audio_pooled_embed], dim=-1)
        #print(out.shape)
        #exit(0)
        
        return self.language_domain_classifier(self.language_GRL(out))

    def task_classify(self, input_ids_out, pixel_values_out,
                      audio_out,
                      pixel_values_attention_mask,
                      input_ids_attention_mask=None):
        
        audio_out = self.audio_FFN(audio_out.last_hidden_state)
        audio_pooled_embed = audio_out.mean(dim=1)
        '''
        out = self.cross_attn(pixel_values_out.last_hidden_state,
                              input_ids_out.pooler_output.unsqueeze(1), # only do it with the pooler ouput since 
                                                                    # we will only use it afterward
                              pixel_values_attention_mask)
        

        
        # residual connection and layer norm
        out = self.cross_attn_layer_norm(out + input_ids_out.pooler_output.unsqueeze(1))
        #print(out.shape)
        #out = self.self_attn(out, input_ids_attention_mask)
        #out = self.self_attn_layer_norm(out + input_ids_out.last_hidden_state)

        out2 = self.cross_attn_FFN(out)
        # residual connection and layer norm
        out2 = self.cross_attn_FFN_layer_norm(out+out2)
        #print(out2.shape)
        #exit(0)
        out2 = torch.cat([out2.squeeze(1), audio_pooled_embed], dim=-1)
        
        '''

        # concat the pooled text embedding with the pooled audio embedding
        #print(out2.shape)
        #exit(0)

        out = self.cross_attn(k=pixel_values_out.last_hidden_state,
                              q=input_ids_out.last_hidden_state,
                              key_attention_mask=pixel_values_attention_mask)
        
        out = self.cross_attn_layer_norm(out + input_ids_out.last_hidden_state)
        out2 = self.cross_attn_FFN(out)
        out2 = self.cross_attn_FFN_layer_norm(out+out2)

        input_ids_attention_mask = input_ids_attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        out2 = out2 * input_ids_attention_mask
        out2 = out2.sum(dim=1)
        lengths = input_ids_attention_mask.sum(dim=1)
        out2 = out2 / lengths.clamp(min=1e-6)

        #out2 = out2.mean(dim=1)
        out2 = torch.cat([out2, audio_pooled_embed], dim=-1)

        return self.task_classifier(out2)

    def forward(self, input_ids, pixel_values, audio,
                #pixel_values_attention_mask=None,
                input_ids_attention_mask=None,
                gender_dc_labels=None,
                language_dc_labels=None,
                labels=None,
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
        # TODO: DONE
            # residual connection and layer norm(RMSnorm?) after attention

        #print("do audio")
        audio_out = self.audio_encoder(audio, return_dict=True)
        #print("###########")
        #print(audio_out.last_hidden_state.shape)
        #exit(0)
        #audio_hidden_states = audio_out.last_hidden_state
        
        #print(audio_out.last_hidden_state.reshape(audio_out.last_hidden_state.shape[0], -1).shape)
        
        audio_pooled_embed = audio_out.last_hidden_state.mean(dim=1)
        #print(f"audio {audio_pooled_embed.shape}")
        #print(audio_pooled_embed.shape)
        #exit(0)
        # feature extracted by convolutional in the model. Contain raw acounstic features
        #audio_feature = audio_out.extract_features

        #print("do text")
        input_ids_out = self.text_encoder(input_ids, input_ids_attention_mask)
        #print(f"text {input_ids_out.pooler_output.shape}")
        #print("do image")
        pixel_values_out = self.vision_encoder(pixel_values)
        #print(f"text {pixel_values_out.last_hidden_state.shape}")
        #print('#############')
        #print(input_ids_out.pooler_output.unsqueeze(1).shape)
        #exit(0)
        
        # zeros means non padding token in pytorch MHA
        pixel_values_attention_mask = torch.zeros((pixel_values_out.last_hidden_state.shape[0],
                                                  pixel_values_out.last_hidden_state.shape[1]),
                                                  dtype=torch.bool, device=pixel_values_out.last_hidden_state.device)
        
        # TODO: DONE
        # only the text pooled emebddings will do the attention

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
        
        #print("do gender dc")
        gender_dc_logits = None
        if gender_dc_labels is not None and gender_dc_labels.dim() > 1: # if training and or given gender_dc_labels(for evaluation)
            #print("33333333333333333333")
            gender_dc_logits = self.gender_domain_classify(audio_pooled_embed)

        #print("do language dc")
        language_dc_logits = None
        if language_dc_labels is not None and language_dc_labels.dim() > 1:
            #print("44444444444444444444")
            language_dc_logits = self.language_domain_classify(input_ids_out=input_ids_out,
                                                               audio_pooled_embed=audio_pooled_embed,
                                                               input_ids_attention_mask=input_ids_attention_mask)

        #print("do task classifier")
        tc_logits = self.task_classify(input_ids_out,
                                       pixel_values_out,
                                       audio_out,
                                       pixel_values_attention_mask,
                                       input_ids_attention_mask)

        gender_dc_loss = torch.tensor(0, device=tc_logits.device)
        language_dc_loss = torch.tensor(0, device=tc_logits.device)
        total_loss = None
        tc_loss = torch.tensor(0, device=tc_logits.device)
        if gender_dc_labels is not None and self.training and gender_dc_labels.dim() > 1:
            #print("111111111111111111111")
            gender_dc_loss = F.binary_cross_entropy_with_logits(gender_dc_logits, gender_dc_labels)

        if language_dc_labels is not None and self.training and language_dc_labels.dim() > 1:
            #print("2222222222222222222222")
            language_dc_loss = F.binary_cross_entropy_with_logits(language_dc_logits, language_dc_labels)
            
        if labels is not None:
            tc_loss = F.binary_cross_entropy_with_logits(tc_logits, labels)

        if gender_dc_labels is not None or labels is not None or language_dc_labels is not None:
            total_loss = tc_loss + self.config.loss_lambda*gender_dc_loss + self.config.loss_lambda*language_dc_loss

        language_dc_loss = None if language_dc_logits is None else language_dc_loss
        gender_dc_loss = None if gender_dc_logits is None else gender_dc_loss

        if not return_dict:
            output = (tc_logits, tc_loss, gender_dc_logits, gender_dc_loss, language_dc_logits, language_dc_loss)
            return ((total_loss,) + output) if total_loss is not None else output
        #print(f"dddddddddd{gender_dc_logits}")
        
        return InvaCogniClassifierOutput(
            loss=total_loss,
            logits=tc_logits,
            #hidden_states=None,
            #attentions=None,
            gender_dc_logits=gender_dc_logits,
            gender_dc_loss=gender_dc_loss,
            language_dc_logits=language_dc_logits,
            language_dc_loss=language_dc_loss,
            tc_loss=tc_loss,
        )


# version with no transformer block
class InvaCogni_no_TB(nn.Module):
    def __init__(self, config, vision_encoder=None, text_encoder=None,
                 audio_encoder=None):
        super().__init__()
        self.config = config # if use PretrainedMoDel as parent class then
                            # don't have to do this

        self.gender_GRL = InvaCogniGRL(config)
        self.language_GRL = InvaCogniGRL(config)
        self.vision_encoder = InvaCogniVisionEncoder(config, vision_encoder)
        self.text_encoder = InvaCogniTextEncoder(config, text_encoder)
        self.audio_encoder = InvacogniAudioEncoder(config, audio_encoder)

        self.audio_FFN = InvaCogniFFN(config, config.audio_FFN)
        self.img_FFN = InvaCogniFFN(config, [[768, 3072], 'gelu', 'dropout-0.3', [3072, 768], 'gelu'])
        self.text_FFN = InvaCogniFFN(config, [[768, 3072], 'gelu', 'dropout-0.3', [3072, 768], 'gelu'])

        self.fuse_FFN1 = InvaCogniFFN(config, [[1536, 3072], 'gelu', 'dropout-0.3', [3072, 768], 'gelu'])
        self.fuse_FFN2 = InvaCogniFFN(config, [[1536, 3072], 'gelu', 'dropout-0.3', [3072, 768], 'gelu'])

        self.gender_domain_classifier = InvaCogniFFN(config, config.gender_domain_classifier_FFN)
        self.language_domain_classifier = InvaCogniFFN(config, config.language_domain_classifier_FFN)
        self.task_classifier = InvaCogniFFN(config, config.task_classifier_FFN)
 
    def gender_domain_classify(self, inp):
        return self.gender_domain_classifier(self.gender_GRL(inp))
    
    def language_domain_classify(self, inp):
        return self.language_domain_classifier(self.language_GRL(inp))


    def forward(self, input_ids, pixel_values, audio,
                #pixel_values_attention_mask=None,
                input_ids_attention_mask=None,
                gender_dc_labels=None,
                language_dc_labels=None,
                labels=None,
                return_dict=True,
                  **kwargs):

        audio_out = self.audio_encoder(audio, return_dict=True)
        audio_out = self.audio_FFN(audio_out.last_hidden_state)
        audio_out = audio_out.mean(dim=1)
        #print(audio_out.shape)
        
        input_ids_out = self.text_encoder(input_ids, input_ids_attention_mask).last_hidden_state
        input_ids_out = self.text_FFN(input_ids_out)

        input_ids_attention_mask = input_ids_attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        input_ids_out = input_ids_out * input_ids_attention_mask
        input_ids_out = input_ids_out.sum(dim=1)
        lengths = input_ids_attention_mask.sum(dim=1)
        input_ids_out = input_ids_out / lengths.clamp(min=1e-6)
        #input_ids_out = input_ids_out.mean(dim=1)
        #print(input_ids_out.shape)

        pixel_values_out = self.vision_encoder(pixel_values).last_hidden_state
        pixel_values_out = self.img_FFN(pixel_values_out)
        pixel_values_out = pixel_values_out.mean(dim=1)
        #print(pixel_values_out.shape)
        #exit(0)
        
        img_text_fused = torch.cat([pixel_values_out, input_ids_out], dim=-1)
        img_text_fused = self.fuse_FFN1(img_text_fused)

        text_audio_fused = torch.cat([audio_out, input_ids_out], dim=-1)
        text_audio_fused = self.fuse_FFN2(text_audio_fused)

        all_fused = torch.cat([img_text_fused, text_audio_fused], dim=-1)

        tc_logits = self.task_classifier(all_fused)

        gender_dc_logits = None
        if gender_dc_labels is not None and gender_dc_labels.dim() > 1: # if training and or given gender_dc_labels(for evaluation)
            gender_dc_logits = self.gender_domain_classify(all_fused)

        language_dc_logits = None
        if language_dc_labels is not None and language_dc_labels.dim() > 1:
            language_dc_logits = self.language_domain_classify(all_fused)

        gender_dc_loss = torch.tensor(0, device=tc_logits.device)
        language_dc_loss = torch.tensor(0, device=tc_logits.device)
        total_loss = None
        tc_loss = torch.tensor(0, device=tc_logits.device)
        if gender_dc_labels is not None and self.training and gender_dc_labels.dim() > 1:
            gender_dc_loss = F.binary_cross_entropy_with_logits(gender_dc_logits, gender_dc_labels)

        if language_dc_labels is not None and self.training and language_dc_labels.dim() > 1:
            language_dc_loss = F.binary_cross_entropy_with_logits(language_dc_logits, language_dc_labels)
            
        if labels is not None:
            tc_loss = F.binary_cross_entropy_with_logits(tc_logits, labels)

        if gender_dc_labels is not None or labels is not None or language_dc_labels is not None:
            total_loss = tc_loss + self.config.loss_lambda*gender_dc_loss + self.config.loss_lambda*language_dc_loss

        language_dc_loss = None if language_dc_logits is None else language_dc_loss
        gender_dc_loss = None if gender_dc_logits is None else gender_dc_loss

        if not return_dict:
            output = (tc_logits, tc_loss, gender_dc_logits, gender_dc_loss, language_dc_logits, language_dc_loss)
            return ((total_loss,) + output) if total_loss is not None else output
        
        return InvaCogniClassifierOutput(
            loss=total_loss,
            logits=tc_logits,
            gender_dc_logits=gender_dc_logits,
            gender_dc_loss=gender_dc_loss,
            language_dc_logits=language_dc_logits,
            language_dc_loss=language_dc_loss,
            tc_loss=tc_loss,
        )

class InvaCogni_2TB(nn.Module):
    def __init__(self, config, vision_encoder=None, text_encoder=None,
                 audio_encoder=None):
        super().__init__()
        self.config = config # if use PretrainedMoDel as parent class then
                            # don't have to do this

        self.gender_GRL = InvaCogniGRL(config)
        self.language_GRL = InvaCogniGRL(config)
        self.vision_encoder = InvaCogniVisionEncoder(config, vision_encoder)
        self.text_encoder = InvaCogniTextEncoder(config, text_encoder)
        self.audio_encoder = InvacogniAudioEncoder(config, audio_encoder)

        self.gender_domain_classifier = InvaCogniFFN(config, config.gender_domain_classifier_FFN)
        self.language_domain_classifier = InvaCogniFFN(config, config.language_domain_classifier_FFN)
        self.task_classifier = InvaCogniFFN(config, config.task_classifier_FFN)
        
        self.cross_attn = InvaCogniCrossAttention(config)
        self.cross_attn_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)
        self.cross_attn_FFN = InvaCogniFFN(config, config.cross_attn_FFN)
        self.cross_attn_FFN_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)

        self.cross_attn2 = InvaCogniAttention2(attention_dropout=config.attention_dropout,
                                               num_attention_heads=config.num_attention_heads,
                                               qdim=512, kdim=768, vdim=768, projdim=512, 
                                               out_features_projvdim=512,
                                               out_features_projoutdim=512,)
        
        self.cross_attn_layer_norm2 = nn.LayerNorm(normalized_shape=512)
        self.cross_attn_FFN2 = InvaCogniFFN(config, [[512, 3072], 'gelu', 'dropout-0.5', [3072, 512], 'gelu'])
        self.cross_attn_FFN_layer_norm2 = nn.LayerNorm(normalized_shape=512)

    def gender_domain_classify(self, inp):
        return self.gender_domain_classifier(self.gender_GRL(inp))
    
    def language_domain_classify(self, inp):
        return self.language_domain_classifier(self.language_GRL(inp))

    def task_classify(self, input_ids_out, pixel_values_out,
                      audio_out,
                      pixel_values_attention_mask,
                      input_ids_attention_mask=None):

        out = self.cross_attn(k=pixel_values_out.last_hidden_state,
                              q=input_ids_out.last_hidden_state,
                              key_attention_mask=pixel_values_attention_mask)
        
        out = self.cross_attn_layer_norm(out + input_ids_out.last_hidden_state)
        out2 = self.cross_attn_FFN(out)
        out2 = self.cross_attn_FFN_layer_norm(out+out2)

        mask = input_ids_attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        out2 = out2 * mask
        out2 = out2.sum(dim=1)
        lengths = mask.sum(dim=1)
        out2 = out2 / lengths.clamp(min=1e-6)
        #out2 = out2.mean(dim=1)
        
        #flipped_mask = ~input_ids_attention_mask
        out = self.cross_attn2(k=input_ids_out.last_hidden_state,
                               v=input_ids_out.last_hidden_state,
                              q=audio_out.last_hidden_state,
                              key_attention_mask=input_ids_attention_mask)
        
        out = self.cross_attn_layer_norm2(out + audio_out.last_hidden_state)
        out3 = self.cross_attn_FFN2(out)
        out3 = self.cross_attn_FFN_layer_norm2(out+out3)
        out3 = out3.mean(dim=1)

        out3 = torch.cat([out2, out3], dim=-1)

        return out3, self.task_classifier(out3)

    def forward(self, input_ids, pixel_values, audio,
                #pixel_values_attention_mask=None,
                input_ids_attention_mask=None,
                gender_dc_labels=None,
                language_dc_labels=None,
                labels=None,
                return_dict=True,
                  **kwargs):

        audio_out = self.audio_encoder(audio, return_dict=True)

        input_ids_out = self.text_encoder(input_ids, input_ids_attention_mask)

        pixel_values_out = self.vision_encoder(pixel_values)

        pixel_values_attention_mask = torch.zeros((pixel_values_out.last_hidden_state.shape[0],
                                                  pixel_values_out.last_hidden_state.shape[1]),
                                                  dtype=torch.bool, device=pixel_values_out.last_hidden_state.device)

        all_fused, tc_logits = self.task_classify(input_ids_out,
                                       pixel_values_out,
                                       audio_out,
                                       pixel_values_attention_mask,
                                       input_ids_attention_mask)
        
        gender_dc_logits = None
        if gender_dc_labels is not None and gender_dc_labels.dim() > 1: # if training and or given gender_dc_labels(for evaluation)
            gender_dc_logits = self.gender_domain_classify(all_fused)

        language_dc_logits = None
        if language_dc_labels is not None and language_dc_labels.dim() > 1:
            language_dc_logits = self.language_domain_classify(all_fused)

        
        gender_dc_loss = torch.tensor(0, device=tc_logits.device)
        language_dc_loss = torch.tensor(0, device=tc_logits.device)
        total_loss = None
        tc_loss = torch.tensor(0, device=tc_logits.device)
        if gender_dc_labels is not None and self.training and gender_dc_labels.dim() > 1:
            gender_dc_loss = F.binary_cross_entropy_with_logits(gender_dc_logits, gender_dc_labels)

        if language_dc_labels is not None and self.training and language_dc_labels.dim() > 1:
            language_dc_loss = F.binary_cross_entropy_with_logits(language_dc_logits, language_dc_labels)
            
        if labels is not None:
            tc_loss = F.binary_cross_entropy_with_logits(tc_logits, labels)

        if gender_dc_labels is not None or labels is not None or language_dc_labels is not None:
            total_loss = tc_loss + self.config.loss_lambda*gender_dc_loss + self.config.loss_lambda*language_dc_loss

        language_dc_loss = None if language_dc_logits is None else language_dc_loss
        gender_dc_loss = None if gender_dc_logits is None else gender_dc_loss

        if not return_dict:
            output = (tc_logits, tc_loss, gender_dc_logits, gender_dc_loss, language_dc_logits, language_dc_loss)
            return ((total_loss,) + output) if total_loss is not None else output
        
        return InvaCogniClassifierOutput(
            loss=total_loss,
            logits=tc_logits,
            #hidden_states=None,
            #attentions=None,
            gender_dc_logits=gender_dc_logits,
            gender_dc_loss=gender_dc_loss,
            language_dc_logits=language_dc_logits,
            language_dc_loss=language_dc_loss,
            tc_loss=tc_loss,
        )