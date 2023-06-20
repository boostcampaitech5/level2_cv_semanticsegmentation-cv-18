import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .swintransformer import SwinTransformer
from .upernet import UPerNet


class SwinTransformerUPerNet(nn.Module):
    def __init__(
        self,
        pretrain_img_size=384,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        out_channels=[128, 256, 512, 1024],
        weights="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
    ) -> None:
        super().__init__()

        encoder = SwinTransformer(
            pretrain_img_size=pretrain_img_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
        )
        encoder.load_state_dict(model_zoo.load_url(weights)["model"], strict=False)
        self.model = UPerNet(29, encoder, out_channels, embed_dim)

    def forward(self, x):
        return self.model(x)
