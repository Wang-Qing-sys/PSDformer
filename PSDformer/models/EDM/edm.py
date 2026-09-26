
import torch.nn as nn

class EncoderDecoderModule(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.encoder=nn.ModuleList([
            nn.Conv1d(dim,dim,3,padding=1),
            nn.Conv1d(dim,dim,3,padding=1)
        ])
        self.decoder=nn.Linear(dim,dim)

    def forward(self,x):
        for e in self.encoder:
            x=e(x)
        return self.decoder(x.transpose(1,2))
