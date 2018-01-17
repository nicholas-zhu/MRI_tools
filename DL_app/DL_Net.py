import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class Net_cnn(nn.Module):
    def __init__(self):
        super(Net_cnn, self).__init__()
        self.conv1 = nn.Conv3d(1,20,5)
        self.conv1_p = nn.Conv3d(1,20,5)
        self.norm1 = nn.BatchNorm3d(20)
        self.conv2 = nn.Conv3d(20,40,5)
        self.norm2 = nn.BatchNorm3d(40)
        self.conv3 = nn.Conv3d(40,20,3)
        self.norm3 = nn.BatchNorm3d(20)
        self.conv4 = nn.Conv3d(20,1,3)
        self.norm4 = nn.BatchNorm3d(1)
        
    def forward(self, x):
        out1 = F.tanh(self.norm1(self.conv1(x)))
        out2 = F.tanhshrink(self.norm1(self.conv1_p(x)))
        out = out1 + out2
        out = F.tanhshrink(self.norm2(self.conv2(out)))
        out = F.tanhshrink(self.norm3(self.conv3(out)))
        out = F.tanhshrink(self.norm4(self.conv4(out)))
        return out
    
from FFT import fft

Phase = np.random.randn(15,15,15)
A = fft.fft(shape=1,axes=(2,3,4))
field_of_view = Phase.shape
yy, xx, zz = np.meshgrid(np.arange(0, Phase.shape[1]),
                         np.arange(0, Phase.shape[0]),
                         np.arange(0, Phase.shape[2]))
xx, yy, zz = ((xx - np.round((Phase.shape[0])/2)) / field_of_view[0],
              (yy - np.round((Phase.shape[1])/2)) / field_of_view[1],
              (zz - np.round((Phase.shape[2])/2)) / field_of_view[2])
k2 = xx**2 + yy**2 + zz**2 + np.spacing(1)
k2 = np.square(xx) + np.square(yy) + np.square(zz)+ np.spacing(1)
k2 = k2[None,None,:]
ik2 = 1/k2
k2.shape


net = Net_cnn()
Iter = 100
optimizer = optim.Adam(net.parameters(), lr=0.01)
optimizer.zero_grad()
criteria = nn.MSELoss()

for _ in range(Iter):
    Phase = np.random.randn(30,1,15,15,15)
    LP = A.IFT(k2*A.FT(Phase)).real
    data, target = Variable(torch.FloatTensor(LP)), Variable(torch.FloatTensor(Phase[:,:,6:-6,6:-6,6:-6]))
    output = net(data)
    loss = criteria(output, target)
    loss.backward()
    optimizer.step()
    print(loss)
    

