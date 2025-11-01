from transformers import PretrainedConfig
from dataclasses import dataclass, field, make_dataclass
import inspect
from typing import Any, List
import ast


class MyModelConfig(PretrainedConfig):
    model_type = "Mymodel" # Must be different from existing model types
    def __init__(self, loss_lambda=0.5,
                 hidden_act="gelu",
                 hidden_size=768,
                 num_domains=4,
                 num_attention_heads=12,
                 audio_FFN="[[768, 3072], 'gelu', [3072, 768], 'gelu']",
                 domain_classifier_FFN="[[768, 3072], 'gelu', [3072, 768], 'gelu', [768, 384], 'gelu', [384, 4]]",
                 task_classifier_FFN="[[1536, 3072], 'gelu', [3072, 768], 'gelu', [768, 384], 'gelu', [384, 1]]",
                 cross_attn_FFN="[[768, 3072], 'gelu', [3072, 768], 'gelu']",
                 grl_lambda=1, # will be turned into negative numnber in the code
                 vision_encoder_path="google/siglip-base-patch16-224",
                 text_encoder_path='FacebookAI/roberta-base',
                 audio_encoder_path='facebook/wav2vec2-base-960h',
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size=hidden_size
        self.hidden_act = hidden_act
        self.loss_lambda = loss_lambda
        self.audio_FFN = ast.literal_eval(audio_FFN)
        
        self.num_domains = num_domains
        self.domain_classifier_FFN = ast.literal_eval(domain_classifier_FFN)
        self.domain_classifier_FFN[-1][-1] = self.num_domains

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