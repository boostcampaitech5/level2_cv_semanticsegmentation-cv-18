import segmentation_models_pytorch as smp
import torch.nn as nn


class FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = smp.FPN("resnet101", encoder_weights="imagenet")

    def forward(self, x):
        return self.model(x)
