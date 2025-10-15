from modeling_mymodel import MyModelCrossAttention, MyModelCrossAttention2

import torch
class config:
    def __init__(self, num_heads, hidden_size):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
# Shape (e.g., 3x4 tensor)
#shape = (3, 4)
img = torch.rand((2, 3, 4))
t = torch.rand((2, 1, 4))
k_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
q_mask = torch.tensor([[1], [1]], dtype=torch.bool)

c = config(2, 4)
#model = MyModelCrossAttention(c)
model = MyModelCrossAttention(c)
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
