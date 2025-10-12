# NEED TO SAMPLE THE AUDIO FILR TO 16000Hz before passing it to the audio model
# processor and the audio model. because they original was train on 16000Hz
# TODO: THE TEXT ENCODER IN SIGLIP AND CLIP ONLY HAVE 77 TOKEN LIMITS. USE BERT?

# TODO: IMPORTANT
# IF USE THE torch.nn.MultiheadAttention then need to flip the 
# attention mask output by the model processor because hugginface 
# 1 and 0 in the attention mask means the opposite

from transformers import ProcessorMixin
import re
from transformers import AutoProcessor
import torch
from urllib.parse import urlparse
from PIL import Image
import requests
import inspect
import gc
from datetime import datetime
import sys
import torch
import torchaudio
#from transformers import Wav2Vec2Processor, Wav2Vec2Model
#import soundfile as sf

class MyModelProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "image_processor", "tokenizer"]

    image_processor_class = "AutoImageProcessor" # for image
    tokenizer_class = "AutoTokenizer" # for text
    feature_extractor_class="AutoFeatureExtractor" # do audio

    def __init__(self, feature_extractor=None, image_processor=None,
        tokenizer=None, **kwargs):
        super().__init__(
                        feature_extractor=feature_extractor,
                        image_processor=image_processor,
                        tokenizer=tokenizer,
                        **kwargs)
    def __call__(self, images, text, audio,
                 dc_labels=None,
                 tc_labels=None,
                 target_sample_rate=16000):
        # Resample if sample_rate != 16kHz (Wav2Vec2 default)
        """
        Do the data preprocessing

        Args:
            audio (list[str]):
                a list of string that are the paths to the audio files
            image (list[str]):
                a list of string that are the paths to the image files
            text (list[str]):
                a list of string that are the paths to the text files

        Returns:
            

        """
        waveforms = []
        for audio_file in audio:
            waveform, sample_rate = torchaudio.load(audio_file)  # waveform shape: (channels(1), num_samples)

            target_sample_rate = 16000 # whisper and wav2vec2 require the audio file being 16000 hz
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveforms.append(resampler(waveform))
        
        waveforms = self.feature_extractor.preprocess(waveforms,
                                                      sampling_rate=target_sample_rate, 
                                                      return_tensors="pt").input_values # return shape: [1, batch size, num channels(1), num_samples]
        waveforms = waveforms.squeeze() # return tensor shape: (batch size, num_samples)
        
        pixel_values = self.image_processor.preprocess(images=[Image.open(path) for path in images], return_tensors="pt").pixel_values
        #pixel_values_attention_mask = 
        input_ids = self.tokenizer(text, truncation=True, padding="longest", return_tensors='pt')
                
        return {
                "audio":waveforms,
                "pixel_values":pixel_values,
                "input_ids":input_ids.input_ids,
                "input_ids_attention_mask":input_ids.attention_mask,
                "dc_labels": dc_labels,
                "tc_labels":tc_labels,
                }

d = MyModelProcessor()
d()
a= 'asd'
a.endswith()