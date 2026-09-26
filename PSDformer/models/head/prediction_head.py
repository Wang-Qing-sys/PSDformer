
import torch.nn as nn

class StressPredictionHead(nn.Module):
    def __init__(self,dim,out=1):
        super().__init__()
        self.fc=nn.Linear(dim,out)

    def forward(self,x):
        return self.fc(x)
