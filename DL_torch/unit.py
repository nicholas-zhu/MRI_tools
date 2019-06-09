import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Conv2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel = 5, padding = 2, stride = 1, drop_out = 0.5, afunc = F.tanh):
        super(Conv2d_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.dropout = nn.Dropout2d(p = drop_out)
        self.afunc = afunc
    
    def forward(self, x):
        #  x:[in x out x D x H x W]
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.afunc(x)
        return x
    
class Conv2d_down(nn.Module):
    def __init__(self, in_ch, out_ch, scale = 2):
        super(Conv2d_down, self).__init__()
        stride = scale
        kernel = (stride-1)*2+1
        padding = stride-1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    
    def forward(self, x):
        #  x:[in x out x D x H]
        x = self.conv1(x)
        return x
    
class Conv2d_up(nn.Module):
    def __init__(self, in_ch, out_ch, scale = 2):
        super(Conv2d_up, self).__init__()
        stride = scale
        kernel = (stride-1)*2+1
        padding = stride-1
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    
    def forward(self, x):
        #  x:[in x out x D x H]
        x = self.conv1(x)
        return x

class Concat2d(nn.Module):
    def __init__(self):
        super(Concat2d, self).__init__()
        
    def forward(self, x1, x2):
        diffD = x1.size()[2] - x2.size()[2]
        diffH = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffD // 2, (diffD+1) // 2,
                        diffH // 2, (diffH+1) // 2))
        x = torch.cat([x2, x1], dim=1)
        return x    

class Conv2d_out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv2d_out, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,3,padding = 1)

    def forward(self, input):
        x = self.conv(input)
        return x    
    
class Conv3d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel = 5, padding = 2, stride = 1, drop_out = 0.5, afunc = F.tanh):
        super(Conv3d_block, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.dropout = nn.Dropout3d(p = drop_out)
        self.afunc = afunc
    
    def forward(self, x):
        #  x:[in x out x D x H x W]
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.afunc(x)
        return x

class Conv3d_down(nn.Module):
    def __init__(self, in_ch, out_ch, scale = 2):
        super(Conv3d_down, self).__init__()
        stride = scale
        kernel = (stride-1)*2+1
        padding = stride-1
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    
    def forward(self, x):
        #  x:[in x out x D x H x W]
        x = self.conv1(x)
        return x
    
class Conv3d_up(nn.Module):
    def __init__(self, in_ch, out_ch, scale = 2):
        super(Conv3d_up, self).__init__()
        stride = scale
        kernel = (stride-1)*2+1
        padding = stride-1
        self.conv1 = nn.ConvTranspose3d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    
    def forward(self, x):
        #  x:[in x out x D x H x W]
        x = self.conv1(x)
        return x

class Concat3d(nn.Module):
    def __init__(self):
        super(Concat3d, self).__init__()
        
    def forward(self, x1, x2):
        diffD = x1.size()[2] - x2.size()[2]
        diffH = x1.size()[3] - x2.size()[3]
        diffW = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffD // 2, (diffD+1) // 2,
                        diffH // 2, (diffH+1) // 2,
                        diffW // 2, (diffW+1) // 2))
        x = torch.cat([x2, x1], dim=1)
        return x    

class Conv3d_out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv3d_out, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Dense_grid(nn.Module):
    def __init__(self):
        super(Dense_grid, self).__init__()

    def forward(self,x,grid):
        # input: [N x C x ID x IH (x IW)]
        # output: [N x ID x IH (x IW) x (2,3)]
        x = F.grid_sample(x,grid)
        return x