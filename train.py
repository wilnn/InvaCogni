from transformers import AutoConfig, AutoModel, BertModel, Wav2Vec2Model
# can not use AutoModel for BERT and Wav2Vec2 because the `architectures`
# in the config specify BertModelForCTC instead of BertModel class
# so if use AutoModel it will load BertModelForCTC model class
# (SAME PROBLEM FOR Wav2Vec2)

# TODO: IMPORTANT (DO THIS IN THE PROCESSOR CLASS OR AFTER PROCESSOR CLASS IN TRAINING LOOP)
        # IF USE THE torch.nn.MultiheadAttention then need to flip the 
        # attention mask output by the model processor because hugginface 
        # 1 and 0 in the attention mask means the opposite

