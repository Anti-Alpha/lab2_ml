import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import evaluate_metrics


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> Tuple:
    test_loss, acc, precision, recall, f1 = evaluate_metrics(
        model, test_loader, loss_function, device
    )

    logging.info(
        f"Test Loss: {test_loss:.4f}, Accuracy: {acc:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}"
    )

    return test_loss, acc, precision, recall, f1