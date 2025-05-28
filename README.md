# 🌸 ml-engineering-lab2

Minimal image classifier on CIFAR-10 using PyTorch and EfficientNetV2.

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

### 2. Run training

```python
import torch
import torch.nn as nn
import torch.optim as optim

from src import download, ingestion, loader, model as mdl, train_model, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    "test_size": 0.2,
    "val_size": 0.2,
    "random_state": 42,
    "lr": 0.001,
    "n_batches": 5,
    "batch_names_select": ["0", "1"]
}

data_dir = download.download_and_extract(
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "./data"
)

train_df, val_df, test_df = ingestion.process_data(data_dir, cfg)
train_loader = loader.create_data_loader(train_df, cfg)
val_loader = loader.create_data_loader(val_df, cfg)
test_loader = loader.create_data_loader(test_df, cfg)

model = mdl.EfficientNetV2(n_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=cfg["lr"])

save_path = "./data/models/best_model.pth"
train_model.train_model(model, train_loader, val_loader, loss_fn, opt, 1, device, save_path=save_path)
test_model.test_model(model, test_loader, loss_fn, device)
```

---

## 🧱 Structure

- `src/download.py` — downloads and extracts CIFAR-10
- `src/ingestion.py` — processes data to train/val/test splits
- `src/loader.py` — creates DataLoaders
- `src/model.py` — includes SimpleNN and EfficientNetV2
- `src/train_model.py` — training loop
- `src/test_model.py` — evaluation metrics

---

## ✅ Features

- Batch-based sampling
- Train/val/test split
- Clean EfficientNetV2 wrapper