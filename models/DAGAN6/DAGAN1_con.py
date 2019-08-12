import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import sys
sys.path.append('/home/xucheng/Code/MRI_tools/')
from DL_torch.network import *
from DL_torch.util import *
from DL_torch.GAN_def import *
import torch.nn.functional as F
from scipy import signal


# patch gan
Datas = np.load('/data/xucheng/TOF/paired_TOF_nonTOF_CT_mask.npy')
Datas = Datas[30:250,...]

def data_sampling(Datas, idxs, slices = 5, crop_size = 160):
    # Datas expected to be 4D [slice, channel, x, y]
    Shiftmx = Datas.shape[2] - crop_size
    Shiftmy = Datas.shape[3] - crop_size
    shiftx = np.random.randint(0,Shiftmx, size=len(idxs)).tolist()
    shifty = np.random.randint(0,Shiftmy, size=len(idxs)).tolist()
    random_scales = (0.01+np.random.rayleigh(size=len(idxs),scale=10))
    random_scales = [np.array([random_scales[i],random_scales[i],1,1])[None,:,None,None] for i in range(len(idxs))]
    data = [(random_scales[i]*Datas[idxs[i]-slices//2:idxs[i]+(slices + 1)//2,:,shiftx[i]:shiftx[i]+crop_size,shifty[i]:shifty[i]+crop_size]) for i in range(len(idxs))]
    return np.asarray(data)

# change output for training purpose
def PET_2_5_sampling(Datas, nbatch, batchSize, slices = 5, crop_size = 160):
    idxs = np.arange(slices//2,Datas.shape[0]-slices//2-1)
    np.random.shuffle(idxs)
    idxs = idxs[:nbatch*batchSize].reshape(nbatch,batchSize).tolist()
    datas = [data_sampling(Datas,idxs[i], slices = slices, crop_size = crop_size) for i in range(nbatch)]
    datas = [{'input':data[:,:,1,:,:],'mask':data[:,:,3,:,:],'CT':data[:,:,2,:,:]*data[:,:,3,:,:], 'output':data[:,:,0,:,:]} for data in datas]
    
    return datas


# test 
class opt: pass
opt.epoch = 1500
opt.n_epochs = 2500
opt.batchSize = 7
opt.nbatch = (Datas.shape[0]-7)//opt.batchSize
opt.lr = 1e-4
opt.decay_epoch = opt.n_epochs//2
opt.input_cn = 7
opt.output_cn = 1
opt.n_cpu = 1
opt.size = 200
opt.cuda = True
opt.device = torch.device("cuda:2") if opt.cuda == True else torch.device("cpu") 
print(opt.device)
slices = opt.input_cn



netG = define_GDA(opt.input_cn, opt.output_cn, 'DADenseCNNGenerator3',resnet_output_idx = opt.input_cn//2)
netD = define_D(opt.output_cn, 64, 'basic')

if opt.cuda:
    netG.to(opt.device)
    netD.to(opt.device)
    
criterion_GAN = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()
criterion_sim5 = SSIM_loss1(window_size = 5)
criterion_sim7 = SSIM_loss1(window_size = 7)
criterion_sim9 = SSIM_loss1(window_size = 9)
criterion_poisson = nn.PoissonNLLLoss(log_input=False)

optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.99))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.99))

checkpoint = torch.load('./netG.pth')
netG.load_state_dict(checkpoint)
checkpoint = torch.load('./optimizer_G.pth')
optimizer_G.load_state_dict(checkpoint)
checkpoint = torch.load('./netD.pth')
netD.load_state_dict(checkpoint)
checkpoint = torch.load('./optimizer_D.pth')
optimizer_D.load_state_dict(checkpoint)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.Tensor
input = Tensor(opt.batchSize, opt.input_cn, opt.size, opt.size).to(opt.device)
input_a = Tensor(opt.batchSize, opt.input_cn, opt.size, opt.size).to(opt.device)
output = Tensor(opt.batchSize, opt.output_cn, opt.size, opt.size).to(opt.device)
ref = Tensor(opt.batchSize, opt.input_cn, opt.size, opt.size).to(opt.device)
mask = Tensor(opt.batchSize, opt.output_cn, opt.size, opt.size).to(opt.device)
target_real = Variable(Tensor(size = [opt.batchSize,opt.output_cn]).fill_(1.0).to(opt.device), requires_grad=False)
target_fake = Variable(Tensor(size = [opt.batchSize,opt.output_cn]).fill_(0.0).to(opt.device), requires_grad=False)

window_size = 5
window1d = signal.gaussian(M = window_size,std = 1)
window = (window1d[None,:]*window1d[:,None]).astype(np.float32)
window = window/window.sum()
gauss_kernel2 =  Variable(torch.from_numpy(window).expand(opt.output_cn, 1, window_size, window_size).contiguous(), requires_grad=False)
if opt.cuda:
    gauss_kernel = gauss_kernel2.to(opt.device)
G_conv = lambda x:F.conv2d(x,gauss_kernel,padding = window_size//2, groups = opt.output_cn)

sobel_kernel = Variable(torch.from_numpy(np.array([[[[1,0,-1],[2,0,-2],[1,0,-1]]],[[[1,2,1],[0,0,0],[-1,-2,-1]]]]).astype(np.float32)).contiguous())
if opt.cuda:
    sobel_kernel = sobel_kernel.to(opt.device)
sobel_conv = lambda x:F.conv2d(x,sobel_kernel, groups = opt.output_cn)

fake_B_buffer = ReplayBuffer()

logger = Logger(opt.n_epochs, opt.nbatch)

for epoch in range(opt.epoch, opt.n_epochs):
    Datas = np.flip(Datas,axis=0)
    dataloader = PET_2_5_sampling(Datas,opt.nbatch,opt.batchSize, slices = slices, crop_size = opt.size)
    for i, batch in enumerate(dataloader):
        real_A = Variable(input.copy_(torch.from_numpy(batch['input'])))
        real_B = Variable(output.copy_(torch.from_numpy(batch['output'][:,slices//2:slices//2+1,:,:])))
        ref_B = Variable(ref.copy_(torch.from_numpy(batch['output'])))
        mask = Variable(mask.copy_(torch.from_numpy(batch['mask'][:,slices//2:slices//2+1,:,:])))
        CT =Variable(input_a.copy_(torch.from_numpy(batch['CT'])))
         ###### Generator ######
        # Similarity and Identity loss
        optimizer_G.zero_grad()
        fake_B,_ = netG(real_A, CT)
        ref_B1,_ = netG(ref_B, CT)
        loss_sim = criterion_sim9(real_B, fake_B, mask)*5.0
        # loss_id2 = criterion_identity(torch.sqrt(fake_B+3/8),torch.sqrt(real_B+3/8))/5
        loss_id2 = criterion_identity(torch.sqrt(fake_B+1e-6),torch.sqrt(real_B+1e-6))
        loss_id = criterion_poisson(fake_B+1, real_B+1)/5
        loss_id3 = criterion_identity(ref_B1, real_B)/5
        pred_fake = netD(fake_B)
        loss_GAN = criterion_GAN(pred_fake, target_real)
        
        loss_G = loss_sim + loss_id + loss_id3 + loss_id2 +loss_GAN
        loss_G.backward()
        
        optimizer_G.step()
        ###################################
        
        ###### Discriminator A ######
        optimizer_D.zero_grad()
        fake_B1 = fake_B.detach()
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        pred_real = netD(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5
        loss_D.backward()

        optimizer_D.step()
        ###################################        
        logger.log({'loss_G': loss_G.cpu(), 'loss_sim': loss_sim.cpu(), 'loss_D': loss_D.cpu()}, 
            images={'real_A': real_A[:,slices//2:slices//2+1,...].cpu(), 'real_B': real_B.cpu(), 'fake_B': fake_B1.cpu(), 'CT':CT[:,slices//2:slices//2+1,...].cpu()})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    # Save models checkpoints
    torch.save(netG.state_dict(), './netG.pth')
    torch.save(netD.state_dict(), './netD.pth')
    torch.save(optimizer_G.state_dict(), './optimizer_G.pth')
    torch.save(optimizer_D.state_dict(), './optimizer_D.pth')
