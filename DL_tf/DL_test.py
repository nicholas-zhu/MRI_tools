import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

data = sio.loadmat(file_name='../qsm_sim')
INDICES = data['INDICES']
qsm_real = data['qsm_real']
TissuePhase = data['TissuePhase']
signal_map = data['signal_map']

# network definition
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

    
net = Net_cnn()
Iter = 100
optimizer = optim.Adam(net.parameters(), lr=0.01)
optimizer.zero_grad()
criteria = nn.MSELoss()
# data prep
width = 7
k_len = width*2+1
mask = np.copy(INDICES)
mask[width:-width,width:-width,width:-width] = 1
x_ind,y_ind,z_ind = np.nonzero(INDICES)

s_ind = np.arange(x_ind.size)
s_size = s_ind.size
batch_N = 5000
N = s_size//batch_N + 1
kernal_ind = np.arange(np.ceil(-width),np.floor(width)+1)
kernal_ind = kernal_ind[None,:]

for i in range(Iter):
    np.random.shuffle(s_ind)
    for k in range(N):
        t_ind = s_ind[k*batch_N:np.minimum((k+1)*batch_N,s_size)]
        x_t = (np.tile(x_ind[t_ind,None],(1,k_len))+kernal_ind).astype(np.int32)
        y_t = (np.tile(y_ind[t_ind,None],(1,k_len))+kernal_ind).astype(np.int32)
        z_t = (np.tile(z_ind[t_ind,None],(1,k_len))+kernal_ind).astype(np.int32)
        x_t = np.tile(x_t[:,:,None,None],(1,1,k_len,k_len)).reshape(-1)
        y_t = np.tile(y_t[:,None,:,None],(1,k_len,1,k_len)).reshape(-1)
        z_t = np.tile(z_t[:,None,None,:],(1,k_len,k_len,1)).reshape(-1)
        
        batch_phase = TissuePhase[x_t,y_t,z_t].reshape([t_ind.size,k_len,k_len,k_len])
        batch_qsm = qsm_real[x_ind[t_ind],y_ind[t_ind],z_ind[t_ind]]
        
        data, target = Variable(torch.FloatTensor(batch_phase[:,None,:,:,:])), Variable(torch.FloatTensor(batch_qsm[None,None,:,None,None]))
        output = net(data)
        loss = criteria(output, target)
        loss.backward()
        optimizer.step()
        print(loss)
net.save_state_dict('training.pt')