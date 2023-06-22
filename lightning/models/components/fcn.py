import torch.nn as nn
from torchvision import models


class FCN(nn.Module):
    def __init__(self, num_classes=29) -> None:
        self.model = models.segmentation.fcn_resnet50(
            weight=models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        )
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]
