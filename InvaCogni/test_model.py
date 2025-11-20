from modeling_invacogni import InvaCogni
from configuration_invacogni import InvaCogniConfig
from processing_invacogni import InvaCogniProcessor
from transformers import AutoTokenizer, RobertaTokenizer, RobertaModel

from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForCTC
import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor
import soundfile as sf


config = InvaCogniConfig()

vision_encoder = AutoModel.from_pretrained(config.vision_encoder_path).vision_model
print("##############")
text_encoder = AutoModel.from_pretrained(config.text_encoder_path)
#print(text_encoder)
#exit(0)
print("##############")

#audio_config = AutoConfig.from_pretrained(config.audio_encoder_path)
#config.mask_time_prob = 0.0 # prevent the model from masking the audio embeddings
#config.mask_feature_prob = 0.0 # prevent the model from masking the audio embeddings
audio_encoder = AutoModel.from_pretrained(config.audio_encoder_path).encoder
#print(type(audio_encoder))
#exit(0)
#print("777777777777777777777777")
#print(audio_encoder.config.mask_feature_prob)
#audio_encoder.train()
#print(audio_encoder.training)
print("##############")

image_processor = AutoImageProcessor.from_pretrained(config.vision_encoder_path)
tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_path)
#print(tokenizer.model_max_length)
#exit(0)
feature_extractor = AutoFeatureExtractor.from_pretrained(config.audio_encoder_path)

processor = InvaCogniProcessor(feature_extractor=feature_extractor,
                             image_processor=image_processor,
                             tokenizer=tokenizer)


model = InvaCogni(config,
                vision_encoder=vision_encoder,
                text_encoder=text_encoder,
                audio_encoder=audio_encoder,
                )
#print(model)
#exit(0)

# print model size in GB
for name, param in model.named_parameters():
    print(name, param.dtype)
    break  # just the first one
total_params = sum(p.numel() for p in model.parameters())  # total number of elements
bytes_per_param = 4  # float32 has 4 bytes
total_bytes = total_params * bytes_per_param
total_gb = total_bytes / (1024 ** 3)
print(f"model size: {total_gb:.1f}GB")



url = "./dataset/images/images/image-1.jpg"
#image = Image.open(requests.get(url, stream=True).raw)
text = "Yes. Do you need me to like zoom in or anything? No. Okay. So I'm going to have you tell me a story with a beginning, middle and an end. Hmm. Okay. There is a group of people who are driving in a car. It looks like it's a while back. Um, it looks like it could even be from an early reader. Um, in the lower picture, um, everybody is calm and just looking out the window. There's a boy and a girl looking out the window and a dog, and there is a man at the wheel driving and a woman who looks like she's asleep with a child next to her in the car between her and the driver. And there are two people in the back seat. It looks like it's probably, a grandmother and a child in the, in the picture above the boy is, um, has his, most of his body out of the window along with the dog. Uh, and it looks a little dangerous and the girl is blowing a big bubble. The, um, it's the other side of the car. So we could see the man, but it's, it looks like it's the same children since they're wearing the same, um, outfits that they're wearing in the other picture. So we could see the other side of the car going in an opposite direction. It may be that they were on their way somewhere. And then it was on their way back home from the, wherever they had gone. And this time the boy in the back seat has also has his head out the window with his hand to his ear. He might even be twins with the boy in the middle, since they have the same color hair look like they're about the same age and they're wearing the same type of shirt. I think that's my story."
text2 = "Replace me by any text you'd likesdf fe eraw awr."
#audio_file = "./1462-170138-0000.wav"
audio_file = "./dataset/taukadial/train/taukdial-002-1.wav"

input = processor(images=[url, url],
                  text=[text, text2],
                  audio=[audio_file, audio_file],
                  tc_labels=[1, 0],
                  gender_dc_labels=[0, 1],
                  language_dc_labels=[1,0],
                  ) # 4 domains total. start from 0 for first domain
print(input)
print(input.keys())
print(input['audio'].shape)
print("bytes:", input['audio'].numel()*input['audio'].element_size(), "≈", input['audio'].numel()*input['audio'].element_size()/1024**2, "MB")
#exit(0)
print(input['input_ids'].shape)
print(input['input_ids_attention_mask'].shape)
print(input['pixel_values'].shape)
print(input['gender_dc_labels'].shape)
print(input['language_dc_labels'].shape)
print(input['tc_labels'].shape)

print("############")
print("Do one forward pass:")

output = model(**input)
print(output)
print("Finished")


