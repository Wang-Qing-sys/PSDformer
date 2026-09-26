
import torch.nn as nn
from .MDSW.mdsw import DimensionSegmentationWindow
from .TDE.tde import TemporalDependencyExtraction
from .head.prediction_head import StressPredictionHead

class PSDformer(nn.Module):
    def __init__(self,variables=5):
        super().__init__()
        self.mdsw=DimensionSegmentationWindow(24,128,variables)
        self.tde=TemporalDependencyExtraction(128)
        self.head=StressPredictionHead(128)

    def forward(self,x):
        x=self.mdsw(x)
        B,L,D,C=x.shape
        x=x.reshape(B,L*D,C)
        x=self.tde(x)
        return self.head(x)
