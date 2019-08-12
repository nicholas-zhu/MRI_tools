import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import sys
sys.path.append('/home/xucheng/Code/MRI_tools/')
import os
from DL_torch.network import *
from DL_torch.util import *
from DL_torch.GAN_def import *
import torch.nn.functional as F
from scipy import signal

def data_sampling(Datas, idxs, slices = 5, crop_size = 160):
    # Datas expected to be 4D [slice, channel, x, y]
    Shiftmx = np.maximum((Datas.shape[2] - crop_size)//2,0)
    Shiftmy = np.maximum((Datas.shape[3] - crop_size)//2,0)

    data = [(Datas[:,idxs[i]-slices//2:idxs[i]+(slices + 1)//2,Shiftmx:-Shiftmx,Shiftmy:-Shiftmy]) for i in range(len(idxs))]
    data = np.asarray(data)
    return data+np.random.uniform(low=-.5, high=.5, size=data.shape)

# change output for training purpose
def SPECT_sampling(Datas, nbatch, batchSize, slices = 5, crop_size = 160):
    idxs = np.arange(slices//2,Datas.shape[1]-slices//2-1)
    np.random.shuffle(idxs)
    idxs = idxs[:nbatch*batchSize].reshape(nbatch,batchSize).tolist()
    datas = [data_sampling(Datas,idxs[i], slices = slices, crop_size = crop_size) for i in range(nbatch)]
    datas = [{'full':data[:,0,:,:,:],'2-fold':data[:,1,:,:,:],'3-fold':data[:,2,:,:,:], '4-fold':data[:,3,:,:,:]} for data in datas]
    
    return datas

class opt: pass
opt.input_cn = 5
opt.output_cn = 1
opt.n_cpu = 1
opt.size = 64
opt.cuda = True
opt.device = torch.device("cuda:2") if opt.cuda == True else torch.device("cpu") 
print(opt.device)
slices = opt.input_cn

opt.epoch = 0
opt.n_epochs = 800
opt.batchSize = 6
opt.nbatch = (36-opt.input_cn)//opt.batchSize
opt.lr = 1e-4
opt.decay_epoch = 200

datapath = '/raid/xucheng/SPECT/nparray/'
datalist = os.listdir(datapath)
datalist.sort()
datalist_train = datalist[0:35]
datalist_valid = datalist[35:]
random.shuffle(datalist_train)
netG = define_G(opt.input_cn, opt.output_cn, 64, 'RDCNN1',resnet_output_idx = opt.input_cn//2)

if opt.cuda:
    netG.to(opt.device)
    
criterion_l2 = torch.nn.MSELoss()
criterion_l1 = torch.nn.SmoothL1Loss()
criterion_sim5 = SSIM_loss1(window_size = 5,non_neg_flag = False)
# optimizer_G = torch.optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)
optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.99))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.Tensor
input = Tensor(opt.batchSize, opt.input_cn, opt.size, opt.size).to(opt.device)
ref = Tensor(opt.batchSize, opt.input_cn, opt.size, opt.size).to(opt.device)
output = Tensor(opt.batchSize, opt.output_cn, opt.size, opt.size).to(opt.device)
mask_t = Tensor(opt.batchSize, opt.output_cn, opt.size, opt.size).to(opt.device)
logger = Logger(opt.n_epochs, opt.nbatch * len(datalist_train))

for epoch in range(opt.epoch, opt.n_epochs):
    for dataname_t in datalist_train:
        Datas = np.load(datapath + dataname_t)
        Datas= Datas[:,70:106,:,:].astype(np.float)
        dataloader = SPECT_sampling(Datas,opt.nbatch,opt.batchSize, slices = slices, crop_size = opt.size)
        for i, batch in enumerate(dataloader):
            real_A = Variable(input.copy_(torch.from_numpy(batch['3-fold']*3)))
            real_B = Variable(output.copy_(torch.from_numpy(batch['full'][:,slices//2:slices//2+1,:,:])))
            ref_B = Variable(ref.copy_(torch.from_numpy(batch['full'])))
            mask = Variable(mask_t.copy_(torch.from_numpy(batch['full'][:,slices//2:slices//2+1,:,:]>5.0)))
             ###### Generator ######
            # Similarity and Identity loss
            optimizer_G.zero_grad()
            fake_B = netG(real_A)
            ref_B1 = netG(ref_B)
            loss_sim = criterion_sim5(real_B, fake_B, mask)*5.0
            loss_id = criterion_l2(fake_B,real_B)
            loss_id2 = criterion_l2(ref_B1,real_B)

            loss_G = loss_sim + loss_id + loss_id2      
            loss_G.backward()

            optimizer_G.step()
            ###################################        
            logger.log({'loss_G': loss_G.cpu(), 'loss_sim': loss_sim.cpu(), 'loss_id': loss_id2.cpu()}, 
                images={'real_A': real_A[:,slices//2:slices//2+1,...].cpu(), 'real_B': real_B.cpu(), 'fake_B': fake_B.cpu(), 'mask':mask.cpu()})

    # Update learning rates
    lr_scheduler_G.step()
    # Save models checkpoints
    torch.save(netG.state_dict(), './netG.pth')
    torch.save(optimizer_G.state_dict(), './optimizer_G.pth')




