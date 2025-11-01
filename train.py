from transformers import AutoModel, Trainer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import HfArgumentParser
from train_args import trainArgs

from my_model.modeling_mymodel import MyModel
from my_model.configuration_mymodel import MyModelConfig
from my_model.processing_mymodel import MyModelProcessor
from transformers import RobertaTokenizer, RobertaModel

from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForCTC
import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor
import soundfile as sf

# TODO: IMPORTANT (DO THIS IN THE PROCESSOR CLASS OR AFTER PROCESSOR CLASS IN TRAINING LOOP)
    # IF USE THE torch.nn.MultiheadAttention then need to flip the 
    # attention mask output by the model processor because hugginface 
    # 1 and 0 in the attention mask means the opposite


class MyMapDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def do_one_fold(X, y, training_args, model_config_args):
    my_model_config = MyModelConfig(**vars(model_config_args))

    vision_encoder = AutoModel.from_pretrained(my_model_config.vision_encoder_path).vision_model
    print("##############")
    
    # TODO: 
        # need to train the self.pooler in the roberta model and freeze the rest
    text_encoder = AutoModel.from_pretrained(my_model_config.text_encoder_path)
    #print(text_encoder)
    #exit(0)

    print("##############")

    config = AutoConfig.from_pretrained(my_model_config.audio_encoder_path)
    config.mask_time_prob = 0.0 # prevent the model from masking the audio embeddings
    audio_encoder = AutoModel.from_pretrained(my_model_config.audio_encoder_path, config=config)
    #print(type(audio_encoder))
    #exit(0)
    print("##############")


    # TODO: 
        # freeze siglip
        # ensure that wav2vec2 is not freeze
        # freeze the roberta encoder but not the pooler layer.
            # we need to train pooler layer

    # TODO:
        # does the model pass to HF trainer need to inherit
        # from PretrainedModel class? because it needs save_pretrained
        # and load pretrained to load and save from checkpoints?

    # TODO:
        # how to only save unfrozen parameters for checkpoints
        # when use HF trainer
        # (after training, can just save the entire model
        # to get the final model)

    model = MyModel(my_model_config,
                    vision_encoder=vision_encoder,
                    text_encoder=text_encoder,
                    audio_encoder=audio_encoder,
                    )

    print(model)

    image_processor = AutoImageProcessor.from_pretrained(my_model_config.vision_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(my_model_config.text_encoder_path)
    #print(tokenizer.model_max_length)
    #exit(0)
    feature_extractor = AutoFeatureExtractor.from_pretrained(my_model_config.audio_encoder_path)

    processor = MyModelProcessor(feature_extractor=feature_extractor,
                                image_processor=image_processor,
                                tokenizer=tokenizer,)

    # TODO: create dataset and collate_fn
        # method 1: pass processor to dataset class and process examples in
        # the __get_item__ method before returning. the collate_fn function
        # will just turn them into batches, get labels out of it
        #  turn them into tensor, etc., (this allow to use multiple worker better)
        # method 2: also do the processor inside the collate_fn function
        # multiple workers don't means much in this case
    
    # TODO: make hf trainer log during training 
        # no val set so no compute metric function needed?
        # maybe use the test set as the val set?


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # TODO: save final model 
    trainer.save_model("./my_custom_model")
    #tokenizer.save_pretrained("./my_custom_model")

    return model
    
def main():
    parser = HfArgumentParser((trainArgs, MyModelConfig.to_dataclass()))

    # Parse arguments from CLI
    training_args, model_config_args = parser.parse_args_into_dataclasses()
    
    # TODO: load dataset (no pytorch dataset yet) and have it as list
    # X = dataset
    # y = the label (should be at the correspodning location)


    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # for each fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = do_one_fold(X, y, training_args, model_config_args)

        # TODO: test model on test split
            # REMEMBER TO USE INFERENCE MODE

    return


if "__name__" == "__main__":
    main()