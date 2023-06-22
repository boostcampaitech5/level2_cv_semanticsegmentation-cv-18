import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .internimage import InternImage
from .upernet import UPerNet


class InternUPerNet(nn.Module):
    def __init__(
        self,
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        out_channels=[64, 128, 256, 512],
        weights="https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth",
    ) -> None:
        super().__init__()

        encoder = InternImage(
            channels=channels,
            depths=depths,
            groups=groups,
        )
        encoder.load_state_dict(model_zoo.load_url(weights)["model"], strict=False)
        self.model = UPerNet(29, encoder, out_channels, channels)

    def forward(self, x):
        return self.model(x)
