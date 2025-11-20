from transformers import PretrainedConfig
from dataclasses import dataclass, field, make_dataclass
import inspect
from typing import Any, List
import ast


class InvaCogniConfig(PretrainedConfig):
    model_type = "InvaCogni" # Must be different from existing model types
    def __init__(self, loss_lambda=0.5,
                 hidden_act="gelu",
                 hidden_size=768,
                 #num_domains=4,
                 num_attention_heads=12,
                 audio_FFN="[[512, 3072], 'gelu', [3072, 768], 'gelu']",
                 gender_domain_classifier_FFN="[[512, 3072], 'gelu', [3072, 512], 'gelu', [512, 1]]",
                 language_domain_classifier_FFN="[[1280, 3072], 'gelu', [3072, 768], 'gelu', [768, 1]]",
                 task_classifier_FFN="[[1536, 3072], 'gelu', [3072, 768], 'gelu', [768, 384], 'gelu', [384, 1]]",
                 cross_attn_FFN="[[768, 3072], 'gelu', [3072, 768], 'gelu']",
                 grl_lambda=1, # will be turned into negative numnber in the code
                 vision_encoder_path="google/siglip-base-patch16-512", # OR use google/siglip-so400m-patch14-384
                 text_encoder_path='google-bert/bert-base-multilingual-uncased', # OR use multilingual roberta
                 audio_encoder_path='openai/whisper-base', # OR use facebook/wav2vec2-large-xlsr-53 or smaller: facebook/wav2vec2-base-960h which is english only. these take too much ram
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size=hidden_size
        self.hidden_act = hidden_act
        self.loss_lambda = loss_lambda
        self.audio_FFN = ast.literal_eval(audio_FFN)
        
        #self.num_domains = num_domains
        self.gender_domain_classifier_FFN = ast.literal_eval(gender_domain_classifier_FFN)
        #self.gender_domain_classifier_FFN[-1][-1] = self.num_domains
        self.language_domain_classifier_FFN = ast.literal_eval(language_domain_classifier_FFN)

        self.task_classifier_FFN=ast.literal_eval(task_classifier_FFN)
        self.cross_attn_FFN = ast.literal_eval(cross_attn_FFN)
        self.grl_lambda = grl_lambda
        self.vision_encoder_path = vision_encoder_path
        self.text_encoder_path = text_encoder_path
        self.audio_encoder_path = audio_encoder_path
    
    @classmethod
    def to_dataclass(cls):
        """
        Create a dataclass with the same fields and defaults as this config class.
        This can be passed to HfArgumentParser for CLI parsing.
        """
        # Inspect the __init__ signature

        signature = inspect.signature(cls.__init__)
        dataclass_fields = []

        for name, param in signature.parameters.items():
            if name in ("self", "**kwargs"):
                continue
            default = param.default if param.default is not inspect.Parameter.empty else None

            # Determine type
            if isinstance(default, list):
                param_type = List[Any]  
                dataclass_fields.append(
                    (name, param_type, field(default_factory=lambda d=default: d))
                )
            else:
                param_type = type(default) if default is not None else Any
                dataclass_fields.append((name, param_type, default))

        ConfigDataclass = make_dataclass(f"{cls.__name__}Dataclass", dataclass_fields)
        return ConfigDataclass