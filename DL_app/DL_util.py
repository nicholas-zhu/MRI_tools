import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

def n_CNN(i_chan,
          o_chan,
          n_layers = 1,
          n_chan = 64,
          kernel_size = 3,
          pad = 1,
          stride = 1,
          dilation = 1,
          bias = True,
          bn = True,
          activation = nn.Tanh()
          ):
    # params prep
    if isinstance(n_chan,list):
        if len(n_chan) == 1:
            n_chan = n_chan * (n_layers - 1)
    elif isinstance(n_chan,int):
        n_chan = [n_chan] * (n_layers - 1)        
    
    # first layer
    cnn3d = [];
    if n_layers == 1:
        layer_1 = nn.Conv3d(i_chan,o_chan,kernel_size = kernel_size,stride = stride,
                      bias = bias, dilation = dilation, padding = pad)
        cnn3d.append(layer_1)
        if bn :
            cnn3d.append(nn.BatchNorm3d(o_chan))
        cnn3d.append(activation)
        return nn.Sequential(*cnn3d)
    else:
        layer_1 = nn.Conv3d(i_chan,n_chan[0],kernel_size = kernel_size,stride = stride,
                      bias = bias, dilation = dilation, padding = pad)
        cnn3d.append(layer_1)
        if bn :
            cnn3d.append(nn.BatchNorm3d(n_chan))
        cnn3d.append(activation)
            
    # mid layers

    for i in range(n_layers-2):
        layer_2 = nn.Conv3d(n_chan[i],n_chan[i+1],kernel_size = kernel_size,stride = stride,
                  bias = bias, dilation = dilation, padding = pad)
        cnn3d.append(layer_2)
        if bn :
            cnn3d.append(nn.BatchNorm3d(n_chan[i+1]))
        cnn3d.append(activation)
            
    # final layer
    layer_3 = nn.Conv3d(n_chan[-1],o_chan,kernel_size = kernel_size,stride = stride,
                      bias = bias, dilation = dilation, padding = pad)
    cnn3d.append(layer_3)
    if bn :
        cnn3d.append(nn.BatchNorm3d(o_chan))
    cnn3d.append(activation)    
    
    return nn.Sequential(*cnn3d)