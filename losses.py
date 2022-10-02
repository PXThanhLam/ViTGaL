import torch
from torch.nn import functional as F
import torch.nn as nn
import math
import numpy as np
class RetrievalLoss(nn.Module):
    def __init__(self, multi_head_loss = False):
        super().__init__()
        self.global_criterion = nn.CrossEntropyLoss() 
        self.local_weight = 1
        self.diver_weight = 0.001
        self.multi_head_loss = multi_head_loss
        

    def forward(self, global_logits, multi_head_attn, mh_local_logits, label):
        global_loss = self.global_criterion(global_logits, label)
#         if not self.multi_head_loss:
#             return global_loss,0,0,0,0
#         nh = multi_head_attn.shape[1]
#         diver_loss = 0
#         for i in range(nh):
#             for j in range(i+1,nh):
#                 hellinger_dist = 1/math.sqrt(2) * torch.norm(torch.sqrt(multi_head_attn[:,i]) - torch.sqrt(multi_head_attn[:,j]), dim = 1)
#                 diver_loss  += 1 - torch.mean(hellinger_dist**2)
#         diver_loss /= nh*(nh-1)
#         total_loss = global_loss + diver_loss*self.diver_weight
        return global_loss,0,0,0,0
        
class ReductionAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_criterion = nn.CrossEntropyLoss() 
        self.ae_weight = 10
        

    def forward(self, global_logits, ae_local, original_local,label):
        global_loss = self.global_criterion(global_logits, label)
        ae_loss = torch.mean((ae_local - original_local)**2)
        total_loss = global_loss + ae_loss*self.ae_weight
        return total_loss,global_loss,ae_loss