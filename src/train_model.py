import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from metrics import evaluate_metrics


def train_model(
    model: nn.Module,
    train: DataLoader,
    val: DataLoader,
    loss_fn: nn.Module,
    opt: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Path = Path("best_model.pth"),
) -> Path:
    model.to(device)
    best = float("inf")
    final_path = None

    for ep in range(num_epochs):
        model.train()
        for x, y in train:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        logging.info(f"Epoch {ep+1}/{num_epochs} - train loss: {loss.item():.4f}")

        val_loss, acc, prec, rec, f1 = evaluate_metrics(model, val, loss_fn, device)
        logging.info(f"Val | loss: {val_loss:.4f}, acc: {acc:.4f}, prec: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

        if val_loss < best:
            best = val_loss
            final_path = save_path
            torch.save(model.state_dict(), final_path)
            logging.info(f"Model saved (val loss: {best:.4f})")

    return final_path