import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetSeg(nn.Module):
    # SegNet network
    def __init__(self, labels):
        super(ResNetSeg, self).__init__()

        trained_resnet = models.resnet50(pretrained=True)

        self.x32 = nn.Sequential(
            *(list(trained_resnet.children())[:-4])
        )

        self.x16 = nn.Sequential(
            *(list(trained_resnet.children())[-4:-3])
        )

        self.x8 = nn.Sequential(
            *(list(trained_resnet.children())[-3:-2])
        )

        self.c32 = nn.Conv2d(512, labels, 1)
        self.c16 = nn.Conv2d(1024, labels, 1)
        self.c8 = nn.Conv2d(2048, labels, 1)

    def forward(self, x):
        original_shape = (x.shape[2], x.shape[3])

        x32 = self.x32(x)
        c32 = self.c32(x32)

        x16 = self.x16(x32)
        c16 = self.c16(x16)

        x8 = self.x8(x16)
        c8 = self.c8(x8)

        c32 = F.upsample_bilinear(c32, original_shape)
        c16 = F.upsample_bilinear(c16, original_shape)
        c8 = F.upsample_bilinear(c8, original_shape)


        x = c32 + c16 + c8
        x = F.log_softmax(x, dim=1)

        return x