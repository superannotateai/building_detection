""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=True, running_on_gpu=True):
        """ 
        Args:
            n_channels - Number of input channels, for us 3,8 or 12.
            n_classes - Number of output classes, for us 1, we classify buildings only.
            bilinear - If we want to use bilinear upsampling or regular.
            running_on_gpu - If running on GPU, must use nn.SyncBatchNorm, otherwise nn.BatchNorm2d.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.n_classes = n_classes

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64, running_on_gpu=running_on_gpu)
        self.down1 = Down(64, 128, running_on_gpu=running_on_gpu)
        self.down2 = Down(128, 256, running_on_gpu=running_on_gpu)
        self.down3 = Down(256, 512 // factor, running_on_gpu=running_on_gpu)
        self.up1 = Up(512, 256, bilinear, running_on_gpu=running_on_gpu)
        self.up2 = Up(256, 128, bilinear, running_on_gpu=running_on_gpu)
        self.up3 = Up(128, 64 * factor, bilinear, running_on_gpu=running_on_gpu)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
