import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Maxpool2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel = 3, padding = 1, stride = 2, drop_out = 0.5, afunc = torch.relu):
        super(Maxpool2d_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel, stride=stride, padding=padding)
        self.dropout_rate = drop_out
        self.dropout = nn.Dropout2d(p = drop_out)
        self.afunc = afunc
    
    def forward(self, x):
        #  x:[in x out x D x H x W]
        x = self.conv1(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.maxpool(x)
        if self.afunc is not None:
            x = self.afunc(x)
        return x

class Conv2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel = 3, padding = 1, stride = 1, drop_out = 0.5, bn_layer = False, afunc = torch.relu):
        super(Conv2d_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.dropout_rate = drop_out
        self.dropout = nn.Dropout2d(p = drop_out)
        self.bn_layer = bn_layer
        if bn_layer:
            self.bn = nn.BatchNorm2d(num_features = out_ch)
        self.afunc = afunc
    
    def forward(self, x):
        #  x:[in x out x D x H x W]
        x = self.conv1(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        if self.afunc is not None:
            x = self.afunc(x)
        if self.bn_layer:
            x = self.bn(x)
        return x
    
class ConvCat2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel = 3, padding = 1, drop_out = 0.5, bn_layer = False, afunc = torch.relu):
        super(ConvCat2d_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=padding)
        self.dropout_rate = drop_out
        self.dropout = nn.Dropout2d(p = drop_out)
        self.bn_layer = bn_layer
        if bn_layer:
            self.bn = nn.BatchNorm2d(num_features = out_ch)
        self.afunc = afunc
    
    def forward(self, input):
        #  x:[in x out x D x H x W]
        if self.bn_layer:
            x = self.bn(input)
        else:
            x = input
        x = self.conv1(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        if self.afunc is not None:
            x = self.afunc(x)
        
        x = torch.cat((input, x),1)
        return x
    
class prea_ConvCat2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel = 3, padding = 1, drop_out = 0.5, bn_layer = False, afunc = torch.relu):
        super(prea_ConvCat2d_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=padding)
        self.dropout_rate = drop_out
        self.dropout = nn.Dropout2d(p = drop_out)
        self.bn_layer = bn_layer
        if bn_layer:
            self.bn = nn.BatchNorm2d(num_features = out_ch)
        self.afunc = afunc
    
    def forward(self, input):
        #  x:[in x out x D x H x W]
        if self.bn_layer:
            x = self.bn(input)
        else:
            x = input
        if self.dropout_rate > 0:
            x = self.dropout(x)
        if self.afunc is not None:
            x = self.afunc(x)
        x = self.conv1(x)
        x = torch.cat((input, x),1)
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
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding, output_padding=1)
    
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
    def __init__(self, in_ch, out_ch, kernel = 3, padding = 1, stride = 1, drop_out = 0.5, afunc = torch.tanh):
        super(Conv3d_block, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.dropout_rate = drop_out
        self.dropout = nn.Dropout3d(p = drop_out)
        self.afunc = afunc
    
    def forward(self, x):
        #  x:[in x out x D x H x W]
        x = self.conv1(x)
        if self.dropout_rate > 0:
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
        self.conv1 = nn.ConvTranspose3d(in_ch, out_ch, kernel, stride=stride, padding=padding, output_padding=1)
    
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
    