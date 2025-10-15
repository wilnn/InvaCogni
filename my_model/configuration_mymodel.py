

class MyModelConfig:
    def __init__(self, loss_lambda=0.5,
                 hidden_act="gelu",
                 hidden_size=768,
                 num_domains=4,
                 num_attention_heads=12,
                 audio_FFN=[[768, 3072], 'gelu', [3072, 768], 'gelu'],
                 domain_classifier_FFN=[[768, 3072], 'gelu', [3072, 768], 'gelu', [768, 384], 'gelu', [384, 4]],
                 task_classifier_FFN=[[768*2, 3072*2], 'gelu', [3072*2, 768], 'gelu', [768, 384], 'gelu', [384, 1]],
                 cross_attn_FFN=[[768, 3072], 'gelu', [3072, 768], 'gelu'],
                 grl_lambda=1, # will be turned into negative numnber in the code
                 vision_encoder_path="google/siglip-base-patch16-224",
                 text_encoder_path='FacebookAI/roberta-base',
                 audio_encoder_path='facebook/wav2vec2-base-960h',
                 **kwargs,
                 ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size=hidden_size
        self.hidden_act = hidden_act
        self.loss_lambda = loss_lambda
        self.audio_FFN = audio_FFN

        domain_classifier_FFN[-1][-1] = num_domains
        self.domain_classifier_FFN = domain_classifier_FFN
        
        self.task_classifier_FFN=task_classifier_FFN
        self.cross_attn_FFN = cross_attn_FFN
        self.grl_lambda = grl_lambda
        self.vision_encoder_path = vision_encoder_path
        self.text_encoder_path = text_encoder_path
        self.audio_encoder_path = audio_encoder_path