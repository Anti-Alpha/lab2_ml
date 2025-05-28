from typing import Any, Dict, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Any] = None):
        self.data = df
        self.aug = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = torch.tensor(self.data.iloc[index]["image"], dtype=torch.float32)
        label = self.data.iloc[index]["label"]
        return self.aug(img), label


def create_data_loader(df: pd.DataFrame, config: Dict[str, Any]) -> DataLoader:
    aug = config.get("transform")
    bs = config.get("batch_size", 32)
    workers = config.get("num_workers", 2)

    ds = ImageDataset(df, transform=aug)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers)