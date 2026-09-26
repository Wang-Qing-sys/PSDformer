
import torch.nn.functional as F

def prediction_loss(pred,target):
    return F.mse_loss(pred,target)
