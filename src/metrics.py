import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0
    targets, outputs = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += loss_function(out, y).item()
            preds = out.argmax(dim=1)
            outputs += preds.cpu().tolist()
            targets += y.cpu().tolist()

    total_loss /= len(val_loader)

    acc = accuracy_score(targets, outputs)
    prec = precision_score(targets, outputs, average="macro", zero_division=0)
    rec = recall_score(targets, outputs, average="macro", zero_division=0)
    f1 = f1_score(targets, outputs, average="macro", zero_division=0)

    return total_loss, acc, prec, rec, f1