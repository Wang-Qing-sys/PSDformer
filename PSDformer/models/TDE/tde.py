
import torch.nn as nn
import torch

class TemporalDependencyExtraction(nn.Module):
    def __init__(self,dim,heads=4):
        super().__init__()
        self.temporal=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.norm=nn.LayerNorm(dim)

    def forward(self,x):
        # temporal dependency
        y,_=self.temporal(x,x,x)
        return self.norm(x+y)
