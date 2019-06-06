import torch
import torch.nn.functional as F

alpha = 0.5

def CrossEntropy2d(input, target):
    """ 2D version of the cross entropy loss """
    return F.binary_cross_entropy(input, target)


def jacard_loss(input, target):
    input = input.view(-1)
    target = target.view(-1)

    inter = input*target
    union = input + target - inter

    return 1 - torch.sum(inter)/torch.sum(union)

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return  ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def bce_dice(input, target):
    input = torch.sigmoid(input)

    return CrossEntropy2d(input, target)+(1-dice_loss(input, target))