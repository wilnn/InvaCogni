from modeling_invacogni import InvaCogniCrossAttention, InvaCogniAttention2

import torch
class config:
    def __init__(self, num_heads, hidden_size):
        self.num_attention_heads = num_heads
        self.hidden_size = hidden_size
        
# Shape (e.g., 3x4 tensor)
#shape = (3, 4)
img = torch.rand((2, 3, 4))
t = torch.rand((2, 1, 4))
k_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
q_mask = torch.tensor([[1], [1]], dtype=torch.bool)

c = config(2, 4)
model2 = InvaCogniAttention2(attention_dropout=0.0, num_attention_heads=2,
                                  qdim=4, kdim=4, vdim=4, projdim=8, 
                                  out_features_projvdim=12, out_features_projoutdim=14)
out = model2(k=img, v=img, q=t, key_attention_mask=k_mask, is_causal=True)
print(out)
print(out.shape)
exit(0)
model = InvaCogniCrossAttention(c)
print("do attention")
#out = model(img, t, k_mask, q_mask)
#print(out)
#print("##############")
k_mask = ~k_mask # flip true and false because pytorch MHA expect that

out, w = model(img, t, k_mask, need_weights=True)
print(out)
print("***********")
print(w)
print("finished")
