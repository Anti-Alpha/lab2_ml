import torch
import torch.nn as nn
from torchvision import models


class SimpleNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.layer1 = nn.Linear(32 * 32 * 3, 128)
        self.layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        return self.layer2(x)


class EfficientNetV2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        net = models.efficientnet_v2_s()
        f = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(f, num_classes)
        self.backbone = net

    def forward(self, x):
        return self.backbone(x)