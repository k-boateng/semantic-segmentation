import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)


class DeepLabV3Wrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output["out"]


def build_deeplabv3(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None

    model = deeplabv3_resnet50(weights=weights)

    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    return DeepLabV3Wrapper(model)