from modeling_mymodel import MyModel
from configuration_mymodel import MyModelConfig
from processing_mymodel import MyModelProcessor
from transformers import RobertaTokenizer, RobertaModel

from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForCTC
import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor
import soundfile as sf


vision_encoder = AutoModel.from_pretrained("google/siglip-base-patch16-224").vision_model
print("##############")
text_encoder = AutoModel.from_pretrained("FacebookAI/roberta-base")
#print(text_encoder)
#exit(0)
print("##############")

config = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")
config.mask_time_prob = 0.0 # prevent the model from masking the audio embeddings
audio_encoder = AutoModel.from_pretrained("facebook/wav2vec2-base-960h", config=config)
#print(type(audio_encoder))
#exit(0)
print("##############")

image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
#print(tokenizer.model_max_length)
#exit(0)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

processor = MyModelProcessor(feature_extractor=feature_extractor,
                             image_processor=image_processor,
                             tokenizer=tokenizer)

config = MyModelConfig()

model = MyModel(config,
                vision_encoder=vision_encoder,
                text_encoder=text_encoder,
                audio_encoder=audio_encoder,
                )
print(model)
#exit(0)
'''
# print model size in GB
for name, param in model.named_parameters():
    print(name, param.dtype)
    break  # just the first one
total_params = sum(p.numel() for p in model.parameters())  # total number of elements
bytes_per_param = 4  # float32 has 4 bytes
total_bytes = total_params * bytes_per_param
total_gb = total_bytes / (1024 ** 3)
print(total_gb)
exit(0)
'''


url = "./image.png"
#image = Image.open(requests.get(url, stream=True).raw)
text = "Replace me by any text you'd like."
text2 = "Replace me by any text you'd likesdf fe eraw awr."
audio_file = "./1462-170138-0000.wav"
input = processor(images=[url, url],
                  text=[text, text2],
                  audio=[audio_file, audio_file],
                  tc_labels=[1, 0],
                  dc_labels=[0, 3]) # 4 domains total. start from 0 for first domain
print(input)
print(input.keys())
print(input['audio'].shape)
print(input['input_ids'].shape)
print(input['input_ids_attention_mask'].shape)
print(input['pixel_values'].shape)
print(input['dc_labels'].shape)
print(input['tc_labels'].shape)

print("############")
print("Do one forward pass:")
output = model(**input)
print(output)
print("Finished")


