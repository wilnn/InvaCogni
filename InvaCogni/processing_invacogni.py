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
import librosa
from transformers import WhisperFeatureExtractor
from transformers import Wav2Vec2FeatureExtractor
import numpy as np

class InvaCogniProcessor(ProcessorMixin):
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
    
    def for_wav2vec2(self, audio, target_sampling_rate):
        if target_sampling_rate != self.feature_extractor.sampling_rate:
            raise ValueError(f"Require the target sampling rate to be {self.feature_extractor.sampling_rate} but get {target_sampling_rate}")
        
        target_sampling_rate = self.feature_extractor.sampling_rate # whisper and wav2vec2 require the audio file being 16000 hz
        waveforms = []
        for audio_file in audio:
            waveform, sampling_rate = torchaudio.load(audio_file)  # waveform shape: (channels(1), num_samples)
            
            # If stereo, convert to mono
            if waveform.shape[0] > 1:
                print("convert to mono")
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sampling_rate != target_sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sampling_rate)
                waveform = resampler(waveform)
                
            waveforms.append(waveform)
        
        waveforms = self.feature_extractor(waveforms,
                                            sampling_rate=target_sampling_rate, 
                                            return_tensors="pt").input_values # return shape: [1, batch size, num channels(1), num_samples]
        
        waveforms = torch.squeeze(waveforms, (0, 2)) # return tensor shape: (batch size, num_samples)
        return waveforms

    def for_whisper(self, audio, target_sampling_rate):
        if target_sampling_rate != self.feature_extractor.sampling_rate:
            raise ValueError(f"Require the target sampling rate to be {self.feature_extractor.sampling_rate} but get {target_sampling_rate}")
        
        target_sampling_rate = self.feature_extractor.sampling_rate # whisper and wav2vec2 require the audio file being 16000 hz
        waveforms = []
        for audio_file in audio:
            waveform, sampling_rate = librosa.load(audio_file, sr=target_sampling_rate, mono=True)
            #waveform, sampling_rate = torchaudio.load(audio_file)  # waveform shape: (channels(1), num_samples)

            waveforms.append(waveform)

        # can make it return the attention mask by return_attention_mask=True
        # because it needs to padd so that the examples in the batch has the 
        # same length but whisper encoder will not use the attention mask
        # because it will simply ignore the silence in the log mel spectrogram
        # which means it will ignore the padding which is just silences automatically
        # refer to: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L631
        waveforms = self.feature_extractor(waveforms,
                                            sampling_rate=target_sampling_rate, 
                                            return_tensors="pt")
        
        return waveforms['input_features'] # shape: [batch size, seq length, dim]

    def __call__(self, images, text, audio,
                 gender_dc_labels=None,
                 language_dc_labels=None,
                 labels=None,
                 target_sampling_rate=16000):
        """
        Do the data preprocessing

        Args:
            audio (list[str]):
                a list of string that are the paths to the audio files
            images (list[str]):
                a list of string that are the paths to the image files
            text (list[str]):
                a list of string that are the text
            labels (list[int]):
            gender_dc_labels (list[int]):

        Returns:
            {
                "audio":waveforms,
                "pixel_values":pixel_values,
                "input_ids":input_ids.input_ids,
                "input_ids_attention_mask":input_ids.attention_mask,
                "gender_dc_labels": gender_dc_labels,
                "labels":labels,
            }
        """
        #print(self.feature_extractor.sampling_rate)
        #exit(0)

        if type(self.feature_extractor) is WhisperFeatureExtractor:
            waveforms = self.for_whisper(audio, target_sampling_rate)
        elif type(self.feature_extractor) is Wav2Vec2FeatureExtractor:
            waveforms = self.for_wav2vec2(audio, target_sampling_rate)
        else:
            raise TypeError("Unsupported audio encoder type")

        #print("111111111111111111111111111")
        #print(images)
        #print(np.array(Image.open(images[0])).shape)
        #exit(0)
        pixel_values = self.image_processor.preprocess(images=[Image.open(path).convert("RGB") for path in images], return_tensors="pt").pixel_values

        input_ids = self.tokenizer(text, truncation=True, padding="longest", return_tensors='pt')
        
        gender_dc_labels = None if not gender_dc_labels else torch.tensor(gender_dc_labels, dtype=torch.float32).unsqueeze(-1)
        language_dc_labels = None if not language_dc_labels else torch.tensor(language_dc_labels, dtype=torch.float32).unsqueeze(-1)
        labels = None if not labels else torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
                "audio":waveforms,
                "pixel_values":pixel_values,
                "input_ids":input_ids.input_ids,
                "input_ids_attention_mask":input_ids.attention_mask,
                "language_dc_labels":language_dc_labels,
                "gender_dc_labels": gender_dc_labels,
                "labels": labels,
                }

