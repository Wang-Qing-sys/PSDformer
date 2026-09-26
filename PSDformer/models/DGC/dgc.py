
import torch.nn as nn

class DenseGraphConvolution(nn.Module):
    def __init__(self,dim,growth=32):
        super().__init__()
        self.conv=nn.Conv1d(dim,dim+growth,3,padding=1)

    def forward(self,x):
        return self.conv(x)
