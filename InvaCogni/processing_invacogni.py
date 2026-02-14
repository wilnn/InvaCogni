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
from torchvision import transforms
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
import random

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
        self.safe_random_augment = transforms.Compose([
                    # small rotation
                    transforms.RandomRotation(degrees=15),  # rotates between -10 and +10
                    # small affine (tiny translate & scale)
                    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.90, 1.10)),
                    # small color jitter — conservative
                    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.4, hue=0.05),
                    # occasionally apply a tiny blur (simulate slight defocus)
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.2),
                    # occasionally convert to grayscale very rarely
                    #transforms.RandomGrayscale(p=0.02),
                ])
    def audio_augment(self, x, sr,
                   p_noise=0.5, snr_range=(12,24),
                   p_shift=0.5, shift_max=0.1,
                   p_gain=0.9, gain_db=(-3,3),
                   p_stretch=0.2, stretch_range=(0.95,1.05)):
        
        def add_noise(x, snr_db=20):
            rms = np.sqrt(np.mean(x**2) + 1e-12)
            rms_n = rms / (10**(snr_db/20))
            noise = np.random.normal(0, rms_n, size=x.shape)
            return x + noise

        def time_shift(x, max_frac=0.2):
            n = len(x)
            shift = int(np.random.uniform(-max_frac, max_frac) * n)
            return np.roll(x, shift)

        def random_gain(x, db_min=-6, db_max=6):
            db = np.random.uniform(db_min, db_max)
            gain = 10 ** (db / 20)
            return x * gain

        def time_stretch_safe(x, rate):
            try:
                return librosa.effects.time_stretch(x.astype(np.float32), rate)
            except Exception:
                return x

        def fix_length(x, target_len):
            cur = len(x)
            if cur == target_len:
                return x
            if cur > target_len:
                start = np.random.randint(0, cur - target_len + 1)
                return x[start:start+target_len]
            else:
                pad = target_len - cur
                left = np.random.randint(0, pad+1)
                right = pad - left
                return np.pad(x, (left, right), mode='constant')
        
        L = len(x)
        y = x.copy()

        if random.random() < p_gain:
            y = random_gain(y, db_min=gain_db[0], db_max=gain_db[1])

        if random.random() < p_shift:
            y = time_shift(y, max_frac=shift_max)

        if random.random() < p_noise:
            snr = float(np.random.uniform(snr_range[0], snr_range[1]))
            y = add_noise(y, snr_db=snr)

        if random.random() < p_stretch:
            rate = float(np.random.uniform(stretch_range[0], stretch_range[1]))
            y = time_stretch_safe(y, rate)

        y = fix_length(y, L)            # keep same length
        # keep values in [-1, 1]
        max_abs = np.max(np.abs(y)) + 1e-12
        if max_abs > 1.0:
            y = y / max_abs
        return y

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

    def for_whisper(self, audio, target_sampling_rate, aug_audio):
        if target_sampling_rate != self.feature_extractor.sampling_rate:
            raise ValueError(f"Require the target sampling rate to be {self.feature_extractor.sampling_rate} but get {target_sampling_rate}")
        
        target_sampling_rate = self.feature_extractor.sampling_rate # whisper and wav2vec2 require the audio file being 16000 hz
        waveforms = []
        for audio_file in audio:
            waveform, sampling_rate = librosa.load(audio_file, sr=target_sampling_rate, mono=True)
            #waveform, sampling_rate = torchaudio.load(audio_file)  # waveform shape: (channels(1), num_samples)
            if aug_audio:
                waveform = self.audio_augment(waveform, sampling_rate)

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
        return waveforms['input_features'] # shape: [batch size, 80, 3000]

    def __call__(self, images, text, audio,
                 gender_dc_labels=None,
                 language_dc_labels=None,
                 labels=None,
                 target_sampling_rate=16000,
                 aug_img=False,
                 aug_audio=False,):
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
            waveforms = self.for_whisper(audio, target_sampling_rate, aug_audio)
        elif type(self.feature_extractor) is Wav2Vec2FeatureExtractor:
            waveforms = self.for_wav2vec2(audio, target_sampling_rate)
        else:
            raise TypeError("Unsupported audio encoder type")

        #print("111111111111111111111111111")
        #print(images)
        #print(np.array(Image.open(images[0])).shape)
        #exit(0)
        for i in range(len(images)):
            images[i] = Image.open(images[i]).convert("RGB")
            if aug_img:
                images[i] = self.safe_random_augment(images[i])
        #pixel_values = self.image_processor.preprocess(images=[Image.open(path).convert("RGB") for path in images], return_tensors="pt").pixel_values
        pixel_values = self.image_processor.preprocess(images=images, return_tensors="pt").pixel_values

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

