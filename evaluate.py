import torch
import torch.nn as nn
import torch.nn.functional as F

class Diceloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=1):
        assert prediction.shape==target.shape,'the dimensions of prediction and target do not match/diceloss'
        intersection=(prediction.view(-1) * target.view(-1)).sum()
        union=prediction.sum()+target.sum()
        dice=(2*intersection+smooth)/(union+smooth)
        dice_loss=1-dice
        return dice_loss

class Dicecoeff(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=1):
        assert prediction.shape == target.shape, 'the dimensions of prediction and target do not match/dice'
        intersection=(prediction.view(-1) * target.view(-1)).sum()
        union=prediction.sum()+target.sum()
        dice=(2*intersection+smooth)/(union+smooth)
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
        intersection=(prediction.view(-1)*target.view(-1)).sum()
        union=target.sum()+prediction.sum()-intersection
        iou=intersection/(union+smooth)
        return iou

def CE(input,target):
    assert input.shape==target.shape,'the dimensions of prediction and target do not match/BCE'
    return F.binary_cross_entropy(input,target)

target=torch.eye(3)
prediction=target

bceloss=CE(prediction,target)
print(bceloss)