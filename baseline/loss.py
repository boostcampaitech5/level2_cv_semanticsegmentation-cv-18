import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def init_loss(name):
    loss_dict = {
        'CE' : nn.CrossEntropyLoss(),
        'BCE' : nn.BCEWithLogitsLoss(),
        'softCE' : smp.losses.SoftCrossEntropyLoss(reduction='mean', smooth_factor=0.1),
        'softBCE' : smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1),
        'dice' : smp.losses.DiceLoss(mode='multilabel', smooth=0.1),
        'tversky' : smp.losses.TverskyLoss(mode='multilabel', smooth=0.1)
        
    }
    return loss_dict[name]





