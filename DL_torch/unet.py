import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from DL.unit import *

class UNet1(nn.Module):
    def __init__(self, n_channels, o_channels, x_channels = 8):
        super(UNet1, self).__init__()
        self.input1 = Conv2d_out(n_channels, x_channels)
        self.conv0 = Conv2d_block(x_channels,x_channels)
        self.down1 = Conv2d_down(x_channels, x_channels*2)
        self.conv1 = Conv2d_block(x_channels*2,x_channels*2)
        self.down2 = Conv2d_down(x_channels*2, x_channels*4)
        self.conv2 = Conv2d_block(x_channels*4,x_channels*4)
        self.down3 = Conv2d_down(x_channels*4, x_channels*8)
        self.conv3 = Conv2d_block(x_channels*8,x_channels*8)
        self.down4 = Conv2d_down(x_channels*8, x_channels*16)
        self.conv4 = Conv2d_block(x_channels*16,x_channels*16)
        self.up1 = Conv2d_up(x_channels*16, x_channels*8)
        self.conv1u = Conv2d_block(x_channels*16,x_channels*8)
        self.up2 = Conv2d_up(x_channels*8, x_channels*4)
        self.conv2u = Conv2d_block(x_channels*8,x_channels*4)
        self.up3 = Conv2d_up(x_channels*4, x_channels*2)
        self.conv3u = Conv2d_block(x_channels*4,x_channels*2)
        self.up4 = Conv2d_up(x_channels*2, x_channels)
        self.conv4u = Conv2d_block(x_channels*2,x_channels)
        self.outc = Conv2d_out(x_channels, o_channels)
        self.concat = Concat2d()
        
    def forward(self, input):
        x0 = self.input1(input)
        x1 = self.conv0(x0)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))
        x = self.conv1u(self.concat(x4,self.up1(x5)))
        x = self.conv2u(self.concat(x3,self.up2(x)))
        x = self.conv3u(self.concat(x2,self.up3(x)))
        x = self.conv4u(self.concat(x1,self.up4(x)))
        x = F.sigmoid(self.outc(x))
        return x
