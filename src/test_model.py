import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics import evaluate_metrics
from typing import Tuple

def test_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple:
    res = evaluate_metrics(model, loader, criterion, device)

    logging.info(
        f"Test | loss: {res[0]:.4f}, acc: {res[1]:.4f}, "
        f"prec: {res[2]:.4f}, recall: {res[3]:.4f}, f1: {res[4]:.4f}"
    )

    return res