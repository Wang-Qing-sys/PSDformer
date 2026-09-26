
from torch.utils.data import Dataset

class DriverStressDataset(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return self.data[i],self.label[i]
