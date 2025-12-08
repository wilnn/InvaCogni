from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from train_args import trainArgs

from InvaCogni.modeling_invacogni import InvaCogni
from InvaCogni.configuration_invacogni import InvaCogniConfig
from InvaCogni.processing_invacogni import InvaCogniProcessor
import unicodedata
#from transformers import RobertaTokenizer, RobertaModel

#from PIL import Image
#import requests
from transformers import (TrainingArguments, TrainerState, TrainerControl,
                          AutoModel, Trainer, HfArgumentParser,
                          AutoTokenizer, AutoFeatureExtractor,
                        AutoProcessor, AutoModel,
                        AutoImageProcessor, AutoModelForCTC,
                        EvalPrediction, TrainerCallback,
                        set_seed, EarlyStoppingCallback)
import torch
#import torchaudio
#from transformers import Wav2Vec2Model, AutoConfig, Wav2Vec2Processor
#import soundfile as sf
import pandas
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
import wandb
import gc
import math
from safetensors.torch import load_file


# TODO: IMPORTANT (DO THIS IN THE PROCESSOR CLASS OR AFTER PROCESSOR CLASS IN TRAINING LOOP)
    # IF USE THE torch.nn.MultiheadAttention then need to flip the 
    # attention mask output by the model processor because hugginface 
    # 1 and 0 in the attention mask means the opposite


# TODO: create dataset and collate_fn
        # method 1: pass processor to dataset class and process examples in
        # the __get_item__ method before returning. the collate_fn function
        # will just turn them into batches, get labels out of it
        #  turn them into tensor, etc., (this allow to use multiple worker better)
        # method 2: also do the processor inside the collate_fn function
        # multiple workers don't means much in this case

    
parser = HfArgumentParser((trainArgs, InvaCogniConfig.to_dataclass()))

# Parse arguments from CLI
training_args, model_config_args = parser.parse_args_into_dataclasses()

class InvaCogniTrainer(Trainer):
    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ) -> torch.Tensor:

        # get the current training step
        current_step = self.state.global_step

        # get the max training step and store it to the custom _max_train_steps attribute
        # this avoid having to call self.get_train_dataloader() every time which will 
        # create the dataloader each time it is being called. 
        if not hasattr(self, "_max_train_steps"):
            if self.args.max_steps and self.args.max_steps > 0:
                self._max_train_steps = self.args.max_steps
            else:
                train_dl = self.get_train_dataloader()
                self._max_train_steps = len(train_dl) * self.args.num_train_epochs
                
        max_steps = max(1, self._max_train_steps)

        # In pytorch DDP, the model is wrapped in another DDP model object and 
        # will actually model be at model.module attribute. 
        wrapped_model = model.module if hasattr(model, "module") else model
        

        #Compute GRL lambda schedule: λ(p) = 2 / (1 + exp(-γ p)) - 1
        gamma = wrapped_model.config.loss_gamma
        #print(f"rrrrrrrrr {self._max_train_steps}")
        p = current_step / max_steps
        lambda_val = 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

        # Update model.config.loss_lambda
        wrapped_model.config.loss_lambda = lambda_val
        #print(f"dddddddddddd {wrapped_model.config.loss_lambda}")

        # Continue the normal HF training step
        return super().training_step(
            model=model,
            inputs=inputs,
            num_items_in_batch=num_items_in_batch
        )

'''
class lossCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl, **kwargs):
'''

'''
def compute_loss(outputs,
                labels,
                num_items_in_batch):
    wandb.log({"tc_loss": outputs['tc_loss'].item()})
    wandb.log({"language_dc_loss": outputs['language_dc_loss'].item()})
    wandb.log({"gender_dc_loss": outputs['gender_dc_loss'].item()})
    return outputs["loss"] if isinstance(outputs, dict) else outputs[0] # copied from the default compute loss function 
'''


class TaukdialDataset(Dataset):
    def __init__(self, ds, processor,
                 audio_parent_path="./dataset/taukadial/train/",
                 image_parent_path="./dataset/images/images/",
                 aug_img=False,
                 aug_audio=False,
                 ):
        
        #self.df = pandas.read_csv(ds_path)
        self.ds = ds
        self.processor = processor

        self.audio_parent_path = audio_parent_path
        self.image_parent_path = image_parent_path

        self.label_map = {"MCI": 1, "NC":0}
        self.gender_map = {"M": 1, "F":0}
        self.language_map = {"english": 1, "chinese":0}
        self.aug_img=aug_img
        self.aug_audio=aug_audio

    def remove_punctuation(self, text):
        return ''.join(
            ch for ch in text
            if not unicodedata.category(ch).startswith('P')
        )
 
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        #row = self.df.iloc[idx]
        row = self.ds[idx]

        audio_file = self.audio_parent_path+row["audio_file"]
        text = row['text'].strip()
        text = self.remove_punctuation(text) if training_args.remove_punc_in_text else text
        image_file = self.image_parent_path+row['image']
        label = self.label_map[row['label']]
        gender = self.gender_map[row['sex']]
        language = self.language_map[row['language']]

        return self.processor(images=[image_file],
                           text=[text],
                           audio=[audio_file],
                           labels=[label],
                           gender_dc_labels=[gender],
                           language_dc_labels=[language],
                           target_sampling_rate=16000,
                           aug_img=self.aug_img,
                           aug_audio=self.aug_audio,
                           )
    
def TaukdialDataset_collate_fn(batch):
    #batch[0]['text_pad_token'] = 0
    #print(batch[1].keys())
    #exit(0)
    '''
    batch is a list of dict
    where each dict is returned by TaukdialDataset.__getitem__()
    '''
    pixel_values = [n['pixel_values'] for n in batch]
    audio = [n['audio'] for n in batch]
    gender_dc_labels = [n['gender_dc_labels'] for n in batch] if training_args.dc_gender else None
    language_dc_labels = [n['language_dc_labels'] for n in batch] if training_args.dc_language else None
    labels = [n['labels'] for n in batch]

    # pad text tokens to the longest
    input_ids = []
    input_ids_attention_mask = []
    max_length = 0
    for n in batch:
        if n['input_ids'].shape[-1] > max_length:
            max_length = n['input_ids'].shape[-1]

    for n in batch:
        to_pad = max_length - n['input_ids'].shape[-1]
        input_ids.append(F.pad(n['input_ids'], (0, to_pad), value=training_args.pad_token))
        input_ids_attention_mask.append(F.pad(n['input_ids_attention_mask'], (0, to_pad), value=0))

    pixel_values = torch.cat(pixel_values, dim=0)
    audio = torch.cat(audio, dim=0)
    input_ids = torch.cat(input_ids, dim=0)
    input_ids_attention_mask = torch.cat(input_ids_attention_mask, dim=0)
    gender_dc_labels = torch.cat(gender_dc_labels, dim=0) if training_args.dc_gender else torch.tensor(-49)
    language_dc_labels = torch.cat(language_dc_labels, dim=0) if training_args.dc_language else torch.tensor(-49)
    labels = torch.cat(labels, dim=0)

    return {
            "audio":audio,
            "pixel_values":pixel_values,
            "input_ids":input_ids,
            "input_ids_attention_mask":input_ids_attention_mask,
            "gender_dc_labels": gender_dc_labels,
            "language_dc_labels": language_dc_labels,
            "labels":labels,
            }

def TaukdialDataset_collate_fn2(batch):
    #batch[0]['text_pad_token'] = 0
    #print(batch[1].keys())
    #exit(0)
    pixel_values = [n['pixel_values'] for n in batch]
    audio = [n['audio'] for n in batch]
    gender_dc_labels = [n['gender_dc_labels'] for n in batch]
    language_dc_labels = [n['language_dc_labels'] for n in batch]
    labels = [n['labels'] for n in batch]

    # pad text tokens to the longest
    input_ids = []
    input_ids_attention_mask = []
    max_length = 0
    for n in batch:
        if n['input_ids'].shape[-1] > max_length:
            max_length = n['input_ids'].shape[-1]
    for n in batch:
        to_pad = max_length - n['input_ids'].shape[-1]
        input_ids.append(F.pad(n['input_ids'], (0, to_pad), value=0))
        input_ids_attention_mask.append(F.pad(n['input_ids_attention_mask'], (0, to_pad), value=0))

    pixel_values = torch.cat(pixel_values, dim=0)
    audio = torch.cat(audio, dim=0)
    input_ids = torch.cat(input_ids, dim=0)
    input_ids_attention_mask = torch.cat(input_ids_attention_mask, dim=0)
    gender_dc_labels = torch.cat(gender_dc_labels, dim=0)
    language_dc_labels = torch.cat(language_dc_labels, dim=0)
    labels = torch.cat(labels, dim=0)

    return {
            "audio":audio,
            "pixel_values":pixel_values,
            "input_ids":input_ids,
            "input_ids_attention_mask":input_ids_attention_mask,
            "gender_dc_labels": gender_dc_labels,
            "language_dc_labels": language_dc_labels,
            "labels":labels,
            }


def compute_metrics_fn(p: EvalPrediction):
    #print(f"eeeeeeeeeeeeeeeee{p.predictions[0]}")
    #print(type(p.predictions))
    #print(len(p.predictions))
    #print(p.predictions[0].shape)
    #print(f"############{p.label_ids[2]}")
    #print(type(p.label_ids))
    #print(len(p.predictions))
    #print(p.label_ids[2].shape)
    #exit(0)
    preds = p.predictions[0].squeeze(-1)
    labels = p.label_ids[2].squeeze(-1).astype(int)
    preds = (preds > 0.5).astype(int)
    tc_f1 = f1_score(labels, preds, average="macro")
    tc_bal_acc = balanced_accuracy_score(labels, preds)
    #print("#########")
    #print(p.label_ids)
    #print(type(p.label_ids))
    #print(len(p.predictions))
    #print(p.label_ids[0].shape)
    #print("#########")
    #print(p.inputs[0])
    #print(type(p.inputs))
    #print(p.inputs.shape)
    #print("#########")
    #print(p.losses)
    #print(type(p.losses))
    #print(p.losses.shape)
    #exit(0)

    return {
        "tc_f1": tc_f1,
        "tc_bal_acc": tc_bal_acc,
        "avg_f1_bal_acc": (tc_f1+tc_bal_acc)/2,
        }


def do_one_fold(train_samples, test_samples, fold_num):

    invacogni_config = InvaCogniConfig(**vars(model_config_args))

    vision_encoder = AutoModel.from_pretrained(invacogni_config.vision_encoder_path).vision_model
    for param in vision_encoder.parameters():
        param.requires_grad = False
    print(f"loaded and froze the vision encoder: {type(vision_encoder)}")
    
    text_encoder = AutoModel.from_pretrained(invacogni_config.text_encoder_path)
    print(f"loaded text encoder: {type(text_encoder)}")
    if not training_args.train_text_encoder and not training_args.dc_language:
        for param in text_encoder.encoder.parameters():
            param.requires_grad = False
        print(f"froze the text encoder (except pooler layer) because train_text_encoder={training_args.train_text_encoder} and training_args.dc_language={training_args.dc_language}")
    

    #config = AutoConfig.from_pretrained(my_model_config.audio_encoder_path)
    #config.mask_time_prob = 0.0 # prevent the model from masking the audio embeddings
    #config.mask_feature_prob = 0.0 # prevent the model from masking the audio embeddings
    #audio_encoder = AutoModel.from_pretrained(my_model_config.audio_encoder_path, config=config)
    audio_encoder = AutoModel.from_pretrained(invacogni_config.audio_encoder_path).encoder
    print(f"loaded audio encoder: {type(audio_encoder)}")
    if not training_args.dc_language and not training_args.dc_gender and not training_args.train_audio_encoder:
        for param in audio_encoder.parameters():
            param.requires_grad = False
        print(f"froze the audio encoder because training_args.dc_language={training_args.dc_language} and training_args.dc_gender={training_args.dc_gender} and training_args.train_audio_encoder={training_args.train_audio_encoder}")
    
    model = InvaCogni(invacogni_config,
                    vision_encoder=vision_encoder,
                    text_encoder=text_encoder,
                    audio_encoder=audio_encoder,
                    )

    #print(model)

    trainer = InvaCogniTrainer(
        model=model,
        args=training_args,
        train_dataset=train_samples,
        eval_dataset=test_samples,
        #tokenizer=tokenizer,
        data_collator=TaukdialDataset_collate_fn,
        #compute_loss_fun=compute_loss,
        compute_metrics=compute_metrics_fn,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=4,
                                         #early_stopping_threshold=0.001,
                                         #)],
    )

    trainer.train()

    # load best model. For some reason, the load best model arg in hf trainer 
    # does not work as it is supposed to. It does not load the best model checkpoint 
    # but still use the model at the end of training.
    # Therefore, need to do this manually (this method of
    # manual loading the best model checkpoint only work if there is only 1 model 
    # checkpoint at that location and that check point is the best model checkpoint
    # (this can be done by using save_total_limit=1 and save_strategy=best))
    if trainer.is_world_process_zero():
        #print("4444444444444444")
        all_items = os.listdir(training_args.output_dir)
        for i in all_items:
            if i.startswith("checkpoint-"):
                #print("333333333333")
                state_dict = load_file(f"{training_args.output_dir}/{i}/model.safetensors")  # returns a dictionary of tensors
                model.load_state_dict(state_dict)
                print(f"loaded model from the checkpoint {training_args.output_dir}/{i}/model.safetensors")
                break
        #trainer.save_model(f"./{training_args.output_dir}")
        #processor.save_pretrained(f"./{training_args.output_dir}")

    return model, trainer.is_world_process_zero()

def evaluate(model, dataset, is_train_dataset, fold_num):

    model.eval()  # Set model to evaluation mode

    test_loader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size,
                                shuffle=False,
                                collate_fn=TaukdialDataset_collate_fn2,)
    
    preds_list = []
    labels_list = []
    genders_list = []
    languages_list = []
    temp_audio_out_list = []
    temp_input_ids_out_list = []

    with torch.inference_mode():  # 2. Enable inference mode
        for batch in tqdm(test_loader):
            batch["audio"] = batch["audio"].to(next(model.parameters()).device)
            batch["pixel_values"] = batch["pixel_values"].to(next(model.parameters()).device)
            batch["input_ids"] = batch["input_ids"].to(next(model.parameters()).device)
            batch["input_ids_attention_mask"] = batch["input_ids_attention_mask"].to(next(model.parameters()).device)
            temp_gen = batch["gender_dc_labels"]
            batch["gender_dc_labels"] = torch.tensor(-49).to(next(model.parameters()).device)
            temp_lang = batch["language_dc_labels"]
            batch["language_dc_labels"] = torch.tensor(-49).to(next(model.parameters()).device)
            batch["labels"] = batch["labels"].to(next(model.parameters()).device)
            
            output = model(**batch)
            #print(output)
            #exit(0)
            
            audio_out = model.audio_encoder(batch['audio'],
                                return_dict=True).last_hidden_state.mean(dim=1)
            input_ids_out = model.text_encoder(batch['input_ids'],
                                        batch['input_ids_attention_mask']).pooler_output

            probs = F.sigmoid(output['logits']).squeeze(-1)
            #print(probs)
            preds = (probs > 0.5).int().cpu().numpy()
            labels = batch['labels'].squeeze(-1).int().cpu().numpy()
            genders = temp_gen.squeeze(-1).int().cpu().numpy()
            languages = temp_lang.squeeze(-1).int().cpu().numpy()
            audio_out = audio_out.cpu().numpy()
            input_ids_out = input_ids_out.cpu().numpy()

            temp_audio_out_list.append(audio_out)
            temp_input_ids_out_list.append(input_ids_out)
            preds_list.append(preds)
            labels_list.append(labels)
            genders_list.append(genders)
            languages_list.append(languages)
        
        # TODO: DONE
            # if > training_args.decision_threshold then class 1 else 0
            # use scikit learn to compute f1 score and
            # balanced accuracy per groups (male, female, english,
            # chinese, MCI, NC)
            # ACCUMUATE THE RESULT ACROSS ALL FOLDS AND TAKE AVERAGE

        # TODO: DONE
            # plot the data reprensetation on 2d graph with T-SNE
            # also write script to plot it before any training
            # plot on the same plot for every fold so that the plots
            # will contains all the datapoints in the dataset at the end
        #print(genders_list[0])
        #print(genders_list[0].shape)
        #print(preds_list)
        #print(preds_list[0].ndim)
        #print(preds_list[3].ndim)
        #exit(0)
        preds_list = np.concatenate(preds_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)
        genders_list = np.concatenate(genders_list, axis=0)
        languages_list = np.concatenate(languages_list, axis=0)
        temp_audio_out_list = np.concatenate(temp_audio_out_list, axis=0)
        temp_input_ids_out_list = np.concatenate(temp_input_ids_out_list, axis=0)
        
        summary = ""

        temp = 'test set' if not is_train_dataset else 'train set'
        # 1 is MCI, 0 is NC
        tc_f1 = f1_score(labels_list, preds_list, average="macro")
        tc_bal_acc = balanced_accuracy_score(labels_list, preds_list)
        
        summary += "General (MCI vs NC) results:\n"
        #print(f"F1 for MCI at fold {fold_num} for {temp}: {f1[1]}")
        #print(f"F1 for NC at fold {fold_num} for {temp}: {f1[0]}")
        summary += f"Average F1 at fold {fold_num} for {temp}: {tc_f1}\n"
        #print(f"Balanced Accuracy for MCI at fold {fold_num} for {temp}: {bal_acc[1]}")
        #print(f"Balanced Accuracy for NC at fold {fold_num} for {temp}: {bal_acc[0]}")
        summary += f"Balanced Accuracy at fold {fold_num} for {temp}: {tc_bal_acc}\n"
        summary += "------------------------------------------\n"

        # for gender. 1 is M, 0 is F
        mask_m = genders_list == 1 # get male samples
        g_preds = preds_list[mask_m]
        g_labels = labels_list[mask_m]
        m_f1 = f1_score(g_labels, g_preds, average="macro")
        m_bal_acc = balanced_accuracy_score(g_labels, g_preds)

        summary += "Gender wise (MCI vs NC) results:\n"
        #print(f"F1 for male MCI at fold {fold_num} for {temp}: {f1[1]}")
        #print(f"F1 for male NC at fold {fold_num} for {temp}: {f1[0]}")
        summary += f"Average F1 for male at fold {fold_num} for {temp}: {m_f1}\n"
        #print(f"Balanced Accuracy for male MCI at fold {fold_num} for {temp}: {bal_acc[1]}")
        #print(f"Balanced Accuracy for male NC at fold {fold_num} for {temp}: {bal_acc[0]}")
        summary += f"Balanced Accuracy for male at fold {fold_num} for {temp}: {m_bal_acc}\n\n"


        mask_f = genders_list == 0 # get female samples
        g_preds = preds_list[mask_f]
        g_labels = labels_list[mask_f]
        f_f1 = f1_score(g_labels, g_preds, average="macro")
        f_bal_acc = balanced_accuracy_score(g_labels, g_preds)
        

        #print(f"F1 for female MCI at fold {fold_num} for {temp}: {f1[1]}")
        #print(f"F1 for female NC at fold {fold_num} for {temp}: {f1[0]}")
        summary += f"Average F1 for female at fold {fold_num} for {temp}: {f_f1}\n"
        #print(f"Balanced Accuracy for female MCI at fold {fold_num} for {temp}: {bal_acc[1]}")
        #print(f"Balanced Accuracy for female NC at fold {fold_num} for {temp}: {bal_acc[0]}")
        summary += f"Average Balanced Accuracy for female at fold {fold_num} for {temp}: {f_bal_acc}\n\n"
        summary += f"Average F1 for all genders at fold {fold_num} for {temp}: {(f_f1+m_f1)/2}\n"
        summary += f"Balanced Accuracy for all genders at fold {fold_num} for {temp}: {(f_bal_acc+m_bal_acc)/2}\n"
        summary += "------------------------------------------\n"


        # for language. 1 is english, 0 is chinese
        mask_e = languages_list == 1 # get english samples
        l_preds = preds_list[mask_e]
        l_labels = labels_list[mask_e]
        e_f1 = f1_score(l_labels, l_preds, average="macro")
        e_bal_acc = balanced_accuracy_score(l_labels, l_preds)
        
        summary += "Language wise (MCI vs NC) results:\n"
        #print(f"F1 for english MCI at fold {fold_num} for {temp}: {f1[1]}")
        #print(f"F1 for english NC at fold {fold_num} for {temp}: {f1[0]}")
        summary += f"Average F1 for english at fold {fold_num} for {temp}: {e_f1}\n"
        #print(f"Balanced Accuracy for english MCI at fold {fold_num} for {temp}: {bal_acc[1]}")
        #print(f"Balanced Accuracy for english NC at fold {fold_num} for {temp}: {bal_acc[0]}")
        summary += f"Balanced Accuracy for english at fold {fold_num} for {temp}: {e_bal_acc}\n\n"

        mask_c = languages_list == 0 # get chinese samples
        l_preds = preds_list[mask_c]
        l_labels = labels_list[mask_c]
        c_f1 = f1_score(l_labels, l_preds, average="macro")
        c_bal_acc = balanced_accuracy_score(l_labels, l_preds)
        
        #print(f"F1 for chinese MCI at fold {fold_num} for {temp}: {f1[1]}")
        #print(f"F1 for chinese NC at fold {fold_num} for {temp}: {f1[0]}")
        summary += f"Average F1 for chinese at fold {fold_num} for {temp}: {c_f1}\n"
        #print(f"Balanced Accuracy for chinese MCI at fold {fold_num} for {temp}: {bal_acc[1]}")
        #print(f"Balanced Accuracy for chinese NC at fold {fold_num} for {temp}: {bal_acc[0]}")
        summary += f"Average Balanced Accuracy for chinese at fold {fold_num} for {temp}: {c_bal_acc}\n\n"
        summary += f"Average F1 for all languages at fold {fold_num} for {temp}: {(e_f1+c_f1)/2}\n"
        summary += f"Average Balanced Accuracy for all languages at fold {fold_num} for {temp}: {(c_bal_acc+e_bal_acc)/2}\n"
        with open(f"./{training_args.output_dir}/result.txt", "a") as f:
            f.write(summary)
    return tc_f1, tc_bal_acc, m_f1, m_bal_acc, f_f1, f_bal_acc, e_f1, e_bal_acc, c_f1, c_bal_acc, temp_audio_out_list, temp_input_ids_out_list, mask_m, mask_f, mask_e, mask_c

if __name__ == "__main__":
    set_seed(training_args.seed)

    image_processor = AutoImageProcessor.from_pretrained(model_config_args.vision_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(model_config_args.text_encoder_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_config_args.audio_encoder_path)
    training_args.pad_token = int(tokenizer.pad_token_id)
    
    processor = InvaCogniProcessor(feature_extractor=feature_extractor,
                                image_processor=image_processor,
                                tokenizer=tokenizer,)

    dataset = pandas.read_csv(training_args.dataset_path)
    val_samples = None
    if training_args.max_dataset_size > 0:
        dataset = dataset.iloc[:training_args.max_dataset_size]
        val_samples = None
    elif training_args.max_dataset_size == 0:
        raise ValueError("the max_dataset_size arg can not be 0")
    else:
        counts = {
            ('english', 'M'): 5,
            ('english', 'F'): 7,
            ('chinese', 'M'): 5,
            ('chinese', 'F'): 6,
        }

        # function that samples up to n rows from each group
        def sample_up_to(g):
            n = counts.get((g.name[0], g.name[1]), 0)
            return g.sample(n=min(n, len(g)), replace=False, random_state=training_args.seed)

        val_samples = (dataset
                    .groupby(['language', 'sex'], group_keys=False)
                    .apply(sample_up_to))
        
        # remaining dataset
        dataset = dataset.drop(index=val_samples.index)
        val_samples = [val_samples.iloc[i] for i in range(len(val_samples))]
        val_samples = TaukdialDataset(val_samples,
                                processor=processor,
                                audio_parent_path=training_args.audio_parent_path,
                                image_parent_path=training_args.image_parent_path,
                                aug_img=False, aug_audio=False,)    

    map_label = {
        "english_M":0,
        "english_F":1,
        "chinese_M":2,
        "chinese_F":3,
    }
    labels = []
    for i in range(len(dataset)):
        labels.append(map_label[f"{dataset.iloc[i]['language']}_{dataset.iloc[i]['sex']}"])
    '''
    temp = {}
    for nn in labels:
        if nn in temp:
            temp[nn] += 1
        else:
            temp[nn] = 1
    print(temp)
    print(len(dataset))
    exit(0)'''
    skf = StratifiedKFold(n_splits=training_args.num_fold, shuffle=True, random_state=training_args.seed)

    os.environ["WANDB_PROJECT"] = training_args.wandb_project_name
    tc_f1 = 0
    tc_bal_acc = 0
    m_f1 = 0
    m_bal_acc = 0
    f_f1 = 0
    f_bal_acc = 0
    e_f1 = 0
    e_bal_acc = 0
    c_f1 = 0
    c_bal_acc = 0

    audio_out_list = []
    input_ids_out_list = []
    mask_m = []
    mask_f = []
    mask_e = []
    mask_c = []

    figures = []
    axes_list = []
    for i in range(3):
        fig, ax = plt.subplots()
        figures.append(fig)
        axes_list.append(ax)

    training_args.run_name = f"{training_args.run_name}_fold_{0}"
    training_args.output_dir = f"{training_args.output_dir}/fold_{0}"
    # for each fold
    for n, (train_index, test_index) in enumerate(skf.split(dataset, labels)):
        training_args.run_name = training_args.run_name[:-1] + str(n)
        training_args.output_dir = training_args.output_dir[:-1] + str(n)
        
        train_samples = [dataset.iloc[n] for n in train_index]
        #train_labels = [dataset.iloc[n]['label'] for n in train_index]
        test_samples = [dataset.iloc[n] for n in test_index]
        #test_labels = [dataset.iloc[n]['label'] for n in test_index]
        
        train_samples = TaukdialDataset(ds=train_samples,
                              processor=processor,
                              audio_parent_path=training_args.audio_parent_path,
                              image_parent_path=training_args.image_parent_path,
                              aug_img=training_args.aug_img,
                              aug_audio=training_args.aug_audio,
                              )
        test_samples = TaukdialDataset(ds=test_samples,
                    processor=processor,
                    audio_parent_path=training_args.audio_parent_path,
                    image_parent_path=training_args.image_parent_path,
                    aug_img=False, aug_audio=False,
                    )
        
        val_samples = val_samples if val_samples is not None else test_samples
        model, is_main_process = do_one_fold(train_samples, val_samples, n)

        if is_main_process:
            os.makedirs(f"./{training_args.output_dir}", exist_ok=True)
            
            # clear prev run
            with open(f"./{training_args.output_dir}/result.txt", "w") as f:
                f.write("")
            print(f"Evaluate model on train set at fold {n}:")
            evaluate(model, train_samples, is_train_dataset=True, fold_num=n)
            with open(f"./{training_args.output_dir}/result.txt", "a") as f:
                f.write("##############################\n")
                f.write("##############################\n")
                f.write("##############################\n")
                f.write("On test fold:\n")
            print(f"Evaluate model on test set at fold {n}:")
            tc_f1t, tc_bal_acct, m_f1t, m_bal_acct, f_f1t, f_bal_acct, e_f1t, e_bal_acct, c_f1t, c_bal_acct, audio_outt, input_ids_outt, mask_mt, mask_ft, mask_et, mask_ct = evaluate(model, test_samples, is_train_dataset=False, fold_num=n)

            tc_f1 += tc_f1t
            tc_bal_acc += tc_bal_acct
            m_f1 += m_f1t
            m_bal_acc += m_bal_acct
            f_f1 += f_f1t
            f_bal_acc += f_bal_acct
            e_f1 += e_f1t
            e_bal_acc += e_bal_acct
            c_f1 += c_f1t
            c_bal_acc += c_bal_acct

            audio_out_list.append(audio_outt)
            input_ids_out_list.append(input_ids_outt)
            mask_m.append(mask_mt)
            mask_f.append(mask_ft)
            mask_e.append(mask_et)
            mask_c.append(mask_ct)
        
        # free up the previous model before creating a new one 
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        report_to = training_args.report_to[0] if isinstance(training_args.report_to, list) else training_args.report_to
        if report_to == "wandb" or report_to == "all":
            wandb.finish()

    if is_main_process:
        tc_f1 /= training_args.num_fold
        tc_bal_acc /= training_args.num_fold
        m_f1 /= training_args.num_fold
        m_bal_acc /= training_args.num_fold
        f_f1 /= training_args.num_fold
        f_bal_acc /= training_args.num_fold
        e_f1 /= training_args.num_fold
        e_bal_acc /= training_args.num_fold
        c_f1 /= training_args.num_fold
        c_bal_acc /= training_args.num_fold
        
        summary = ""
        
        with open(f"./{training_args.output_dir[:-7]}/avg_result.txt", "w") as f:
            summary += "General (MCI vs NC) results:\n"
            #print(f"F1 for MCI: {tc_f1[1]}")
            #print(f"F1 for NC: {tc_f1[0]}")
            summary +=f"average F1: {tc_f1}\n"
            #print(f"Balanced Accuracy for MCI: {tc_bal_acc[1]}")
            #print(f"Balanced Accuracy for NC: {tc_bal_acc[0]}")
            summary += f"average Balanced Accuracy: {tc_bal_acc}\n"
            summary +="------------------------------------------\n"
            summary +="Gender wise (MCI vs NC) results:\n"
            #print(f"F1 for male MCI: {m_f1[1]}")
            #print(f"F1 for male NC: {m_f1[0]}")
            summary +=f"Average F1 for male: {m_f1}\n"
            #print(f"Balanced Accuracy for male MCI: {m_bal_acc[1]}")
            #print(f"Balanced Accuracy for male NC: {m_bal_acc[0]}")
            summary +=f"Average Balanced Accuracy for male: {m_bal_acc}\n"
            
            #print(f"F1 for female MCI: {f_f1[1]}")
            #print(f"F1 for female NC: {f_f1[0]}")
            summary +=f"Average F1 for female: {f_f1}\n"
            #print(f"Balanced Accuracy for female MCI: {f_bal_acc[1]}")
            #print(f"Balanced Accuracy for female NC: {f_bal_acc[0]}")
            summary +=f"Average Balanced Accuracy for female: {f_bal_acc}\n"
            summary +=f"Average F1 for all genders: {(m_f1+f_f1)/2}\n"
            summary +=f"Average Balanced Accuracy for all genders: {(m_bal_acc+f_bal_acc)/2}\n"
            summary +="------------------------------------------\n"
            summary +="Language wise (MCI vs NC) results:\n"
            #print(f"F1 for english MCI: {e_f1[1]}")
            #print(f"F1 for english NC: {e_f1[0]}")
            summary +=f"Average F1 for english: {e_f1}\n"
            #print(f"Balanced Accuracy for english MCI: {e_bal_acc[1]}")
            #print(f"Balanced Accuracy for english NC: {e_bal_acc[0]}")
            summary +=f"Average Balanced Accuracy for english: {e_bal_acc}\n"
            
            #print(f"F1 for chinese MCI: {c_f1[1]}")
            #print(f"F1 for chinese NC: {c_f1[0]}")
            summary +=f"Average F1 for chinese: {c_f1}\n"
            #print(f"Balanced Accuracy for chinese MCI: {c_bal_acc[1]}")
            #print(f"Balanced Accuracy for chinese NC: {c_bal_acc[0]}")
            summary +=f"Average Balanced Accuracy for chinese: {c_bal_acc}\n"
            summary +=f"Average F1 for all languages: {(e_f1+c_f1)/2}\n"
            summary +=f"Average Balanced Accuracy for all languages: {(e_bal_acc+c_bal_acc)/2}\n"
            f.write(summary)
        
        audio_out_list = np.concatenate(audio_out_list, axis=0)
        input_ids_out_list = np.concatenate(input_ids_out_list, axis=0)
        mask_m = np.concatenate(mask_m, axis=0)
        mask_f = np.concatenate(mask_f, axis=0)
        mask_e = np.concatenate(mask_e, axis=0)
        mask_c = np.concatenate(mask_c, axis=0)

        perx = math.ceil(audio_out_list.shape[0]/2) if audio_out_list.shape[0] <= 30 else 30
        tsne = TSNE(n_components=2,
                perplexity=perx,
                random_state=training_args.seed)
        # turn into 2D points to plot
        audio_out_list = tsne.fit_transform(audio_out_list)
        input_ids_out_list = tsne.fit_transform(input_ids_out_list)

        axes_list[0].scatter(audio_out_list[mask_m][:,0], audio_out_list[mask_m][:,1], color='orange', label='Male', alpha=0.5)
        axes_list[0].scatter(audio_out_list[mask_f][:,0], audio_out_list[mask_f][:,1],color='blue', label='Female', alpha=0.5)
        axes_list[1].scatter(audio_out_list[mask_e][:,0], audio_out_list[mask_e][:,1],color='orange', label='English', alpha=0.5)
        axes_list[2].scatter(input_ids_out_list[mask_e][:,0], input_ids_out_list[mask_e][:,1],color='orange', label='English', alpha=0.5)
        axes_list[1].scatter(audio_out_list[mask_c][:,0], audio_out_list[mask_c][:,1],color='blue', label='Chinese', alpha=0.5)
        axes_list[2].scatter(input_ids_out_list[mask_c][:,0], input_ids_out_list[mask_c][:,1],color='blue', label='Chinese', alpha=0.5)

        axes_list[0].legend()
        axes_list[0].set_title(f'Male vs Female Audio Embeddings') 
        figures[0].savefig(f"./{training_args.output_dir[:-7]}/m_vs_f_audio.png")
        axes_list[1].legend()
        axes_list[1].set_title(f'English vs Chinese Audio Embeddings') 
        figures[1].savefig(f"./{training_args.output_dir[:-7]}/en_vs_cn_audio.png")
        axes_list[2].legend()
        axes_list[2].set_title(f'English vs Chinese Text Embeddings') 
        figures[2].savefig(f"./{training_args.output_dir[:-7]}/en_vs_cn_text.png")