from typing import Any, Dict, Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import logging
logging.basicConfig(level=logging.INFO)

class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Any] = None):
        self.df = dataframe
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = torch.tensor(self.df.iloc[idx]["image"], dtype=torch.float32)
        label = self.df.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_data_loader(df: pd.DataFrame, config: Dict[str, Any]) -> DataLoader:
    return DataLoader(
        ImageDataset(df, config.get("transform")),
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        num_workers=config.get("num_workers", 2)
    )