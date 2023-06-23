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
        'dice' : smp.losses.DiceLoss(mode='multilabel', smooth=0.1), #DiceLoss(), 
        'dicefocal' : DiceFocalLoss(0.9, 0.1),
        'tversky' : smp.losses.TverskyLoss(mode='multilabel', smooth=0.1)
        
    }
    return loss_dict[name]




from numpy.lib.arraysetops import union1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class custom_CrossEntropyLoss(nn.Module):

    def __init__(self, weights):
        nn.Module.__init__(self)
        self.CEL = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())


    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0
            weights = [1, 0.2]
            assert len(weights) == len(pred)
            for i in range(len(pred)):
                loss += self.CEL(pred[i], target) * weights[i]
            return loss

        else:
            return self.CEL(pred, target)
          
          
class mIoULoss(nn.Module):
    """
    code reference: https://discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152
    """
    def __init__(self, weight=None, size_average=True, n_classes=11):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, pred, target, smooth = 1e-6):
        """
        pred: y_pred (N,C,H,W)
        target: y_true (should be scattered into N,C,H,W shaped tensor)
        """

        N = pred.size()[0]

        pred = F.softmax(pred, dim = 1)
        target_one_hot = self._to_one_hot(target)

        # intersection (numerator)
        intersec = pred * target_one_hot
        intersec = intersec.view(N, self.classes, -1).sum(2)  # sum over all pixels NxCxHxW => NxC

        # union (denominator)
        union = pred + target_one_hot - (pred*target_one_hot)
        union = union.view(N,self.classes,-1).sum(2)

        loss = (intersec+smooth)/(union+smooth)

        return -loss.mean() # miou는 최대화 문제이므로 최소화로 문제를 바꿔서 생각해줘야.

    
    def _to_one_hot(self, target):
        n,h,w = target.size()
        one_hot = torch.zeros(n,self.classes,h,w).cuda().scatter_(1, target.view(n,1,h,w), 1)
        return one_hot


class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()
        self.DL = smp.losses.DiceLoss(mode = "multilabel")

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0
            weights = [1, 0.2]
            assert len(weights) == len(pred)
            for i in range(len(pred)):
                loss += self.DL(pred[i], target) * weights[i]
            return loss

        else:
            return self.DL(pred, target)


class DiceCELoss(nn.Module):

    def __init__(self, dice_weight, ce_weight):

        super().__init__()
        self.DL = DiceLoss()
        self.CEL = custom_CrossEntropyLoss()
        self.weights = [dice_weight, ce_weight]

    def forward(self, pred, target):
        return self.weights[0] * self.DL(pred, target) + self.weights[1] * self.CEL(pred, target)


class FocalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.FL = smp.losses.FocalLoss(mode = "multilabel")

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0
            weights = [1, 0.2]
            assert len(weights) == len(pred)
            for i in range(len(pred)):
                loss += self.FL(pred[i], target) * weights[i]
            return loss

        else:
            return self.FL(pred, target)


class DiceFocalLoss(nn.Module):

    def __init__(self, dice_weight, focal_weight):

        super(DiceFocalLoss, self).__init__()
        self.DL = DiceLoss()
        self.FL = FocalLoss()
        self.weights = [dice_weight, focal_weight]

    def forward(self, pred, target):
        return self.weights[0]*self.DL(pred, target) + self.weights[1] * self.FL(pred, target)