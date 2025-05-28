import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss += loss_function(val_outputs, val_targets).item()
            preds = torch.argmax(val_outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(val_targets.cpu().numpy())

    val_loss /= len(val_loader)
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, acc, precision, recall, f1