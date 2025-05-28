from typing import Any, Dict, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Any] = None):
        self.data = df
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.iloc[index]["image"]
        label = self.data.iloc[index]["label"]

        # image is already a tensor (float32), skip ToTensor
        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label