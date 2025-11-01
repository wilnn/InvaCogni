# NEED TO SAMPLE THE AUDIO FILR TO 16000Hz before passing it to the audio model
# processor and the audio model. because they original was train on 16000Hz

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
            images (list[str]):
                a list of string that are the paths to the image files
            text (list[str]):
                a list of string that are the text
            tc_labels (list[int]):
            dc_labels (list[int]):

        Returns:
            {
                "audio":waveforms,
                "pixel_values":pixel_values,
                "input_ids":input_ids.input_ids,
                "input_ids_attention_mask":input_ids.attention_mask,
                "dc_labels": dc_labels,
                "tc_labels":tc_labels,
            }
        """
        waveforms = []
        target_sample_rate = 16000 # whisper and wav2vec2 require the audio file being 16000 hz
        for audio_file in audio:
            waveform, sample_rate = torchaudio.load(audio_file)  # waveform shape: (channels(1), num_samples)
            
            # If stereo, convert to mono
            if waveform.shape[0] > 1:
                print("convert to mono")
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
            
            waveforms.append(waveform)
        
        waveforms = self.feature_extractor(waveforms,
                                            sampling_rate=target_sample_rate, 
                                            return_tensors="pt").input_values # return shape: [1, batch size, num channels(1), num_samples]
        
        waveforms = torch.squeeze(waveforms, (0, 2)) # return tensor shape: (batch size, num_samples)
        
        pixel_values = self.image_processor.preprocess(images=[Image.open(path) for path in images], return_tensors="pt").pixel_values

        input_ids = self.tokenizer(text, truncation=True, padding="longest", return_tensors='pt')
        
        dc_labels = None if not dc_labels else torch.tensor(dc_labels, dtype=torch.long)
        tc_labels = None if not tc_labels else torch.tensor(tc_labels, dtype=torch.float32).unsqueeze(-1)

        return {
                "audio":waveforms,
                "pixel_values":pixel_values,
                "input_ids":input_ids.input_ids,
                "input_ids_attention_mask":input_ids.attention_mask,
                "dc_labels": dc_labels,
                "tc_labels":tc_labels,
                }

