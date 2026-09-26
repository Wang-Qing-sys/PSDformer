
import torch
import torch.nn as nn

class DimensionSegmentationWindow(nn.Module):
    def __init__(self, segment_length, embed_dim, variables):
        super().__init__()
        self.segment_length=segment_length
        self.proj=nn.Linear(segment_length, embed_dim)
        self.variables=variables

    def forward(self,x):
        # x: batch,time,dimension
        B,T,D=x.shape
        tokens=[]
        for d in range(D):
            xd=x[:,:,d]
            seg=xd.unfold(1,self.segment_length,self.segment_length)
            tokens.append(self.proj(seg))
        return torch.stack(tokens,dim=2)
