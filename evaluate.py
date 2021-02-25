import torch
import torch.nn as nn
import torch.nn.functional as F

class Diceloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=1e-12):
        assert prediction.shape==target.shape,'the dimensions of prediction and target do not match/diceloss'
        batch_size=prediction.shape[0]
        intersection=(prediction.view(batch_size,-1) * target.view(batch_size,-1)).sum(dim=1)
        union=prediction.view(batch_size,-1).sum(dim=1)+target.view(batch_size,-1).sum(dim=1)
        dice=(2*intersection+smooth)/(union+smooth)
        dice_loss = 1 - dice
        dice_loss=torch.mean(dice_loss)
        return dice_loss

class Dicecoeff(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=1e-12):
        assert prediction.shape == target.shape, 'the dimensions of prediction and target do not match/dice'
        batch_size = prediction.shape[0]
        intersection = (prediction.view(batch_size, -1) * target.view(batch_size, -1)).sum(dim=1)
        union = prediction.view(batch_size, -1).sum(dim=1) + target.view(batch_size, -1).sum(dim=1)
        dice = (2 * intersection + smooth) / (union + smooth)
        dice = torch.mean(dice)
        return dice

class Iou(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=0.00001):
        assert prediction.shape == target.shape, 'the dimensions of prediction and target do not match/iou'
        mask=prediction<0.5
        prediction[mask]=0#小于0.5的预测为False,在prediction——copy里小于0.5的被置为0
        mask=prediction>=0.5
        prediction[mask]=1
        batch_size = prediction.shape[0]
        intersection=(prediction.view(batch_size,-1) * target.view(batch_size,-1)).sum(dim=1)
        union=target.view(batch_size, -1).sum(dim=1)+prediction.view(batch_size, -1).sum(dim=1)-intersection
        iou=intersection/(union+smooth)
        iou=torch.mean(iou)
        return iou

def CE(input,target):
    assert input.shape==target.shape,'the dimensions of prediction and target do not match/BCE'
    return F.binary_cross_entropy(input,target)

# target=torch.eye(3)
# prediction=target
#
# bceloss=CE(prediction,target)
# print(bceloss)