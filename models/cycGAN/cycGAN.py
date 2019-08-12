import torch
import itertools
import torch.nn as nn
from torch.nn import init
from torch import Tensor
from torch.autograd import Variable
from DL_torch.network import *
from DL_torch.util import *
import torch.nn.functional as F
import numpy as np
from scipy import signal

data_folder = '/data/xucheng/TOF/'
ToF_img = np.load(data_folder+'TOF_slices.npy')
non_ToF_img = np.load(data_folder+'non_TOF_slices.npy')
N = 6000
batchSize = 10
np.random.shuffle(ToF_img)
np.random.shuffle(non_ToF_img)
crop_size = 48
dataloader = [{'A':ToF_img[i:i+batchSize,None,crop_size:-crop_size,crop_size:-crop_size],'B':non_ToF_img[i:i+batchSize,None,crop_size:-crop_size,crop_size:-crop_size]} for i in range(0,N,batchSize)]
batches = len(dataloader)

# test 
class opt: pass
opt.epoch = 0
opt.n_epochs = 100
opt.batchSize = batchSize
opt.lr = 1e-4
opt.decay_epoch = 20
opt.input_nc = 1
opt.output_nc = 1
opt.n_cpu = 1
opt.size = 256-2*crop_size
opt.cuda = True

# netG_A2B = define_G(opt.input_nc,opt.input_nc,16,'ResCNN')
# netG_B2A = define_G(opt.input_nc,opt.input_nc,16,'ResCNN')
netG_share = define_G_sub(opt.input_nc,8,'CNNModule_share')
netG_A2Bs = define_G_sub(8,opt.input_nc,'CNNModule_split')
netG_B2As = define_G_sub(8,opt.input_nc,'CNNModule_split')
netG_res = define_G_sub(1,1,'ResNetModule')
netG_A2B = lambda x:netG_res(x,nn.Sequential(netG_share,netG_A2Bs))
netG_B2A = lambda x:netG_res(x,nn.Sequential(netG_share,netG_B2As))
netD_A = define_D(opt.input_nc, 64, 'basic')
netD_B = define_D(opt.input_nc, 64, 'basic')
if opt.cuda:
    netG_A2Bs.cuda()
    netG_B2As.cuda()
    netG_share.cuda()
    netD_A.cuda()
    netD_B.cuda()

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
# optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
#                                 lr=opt.lr, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(itertools.chain(netG_share.parameters(), netG_A2Bs.parameters(), netG_B2As.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
window_size = 7
window1d = signal.gaussian(M = window_size,std = 2)
window = (window1d[None,:]*window1d[:,None]).astype(np.float32)
gauss_kernel2 =  Variable(torch.from_numpy(window).expand(opt.input_nc, 1, window_size, window_size).contiguous(), requires_grad=False)
if opt.cuda:
    gauss_kernel = gauss_kernel2.cuda()
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
G_conv = lambda x:F.conv2d(x,gauss_kernel,padding = window_size//2, groups = opt.input_nc)

logger = Logger(opt.n_epochs, len(dataloader))
        
for epoch in range(opt.epoch, opt.n_epochs):
    np.random.shuffle(ToF_img)
    np.random.shuffle(non_ToF_img)
    crop_size = 48
    dataloader = [{'A':ToF_img[i:i+batchSize,None,crop_size:-crop_size,crop_size:-crop_size],'B':non_ToF_img[i:i+batchSize,None,crop_size:-crop_size,crop_size:-crop_size]} for i in range(0,N,batchSize)]
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(torch.from_numpy(batch['A'])))
        real_B = Variable(input_B.copy_(torch.from_numpy(batch['B'])))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # intensity preserve 
        loss_ip_A = criterion_cycle(G_conv(fake_B), G_conv(real_A))*10.0
        loss_ip_B = criterion_cycle(G_conv(fake_A), G_conv(real_B))*10.0
        
        # Total loss
        loss_G = loss_ip_A + loss_ip_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        # loss_G =  loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G.cpu(), 'loss_G_identity': (loss_identity_A + loss_identity_B).cpu(), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).cpu(),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).cpu(), 'loss_D': (loss_D_A + loss_D_B).cpu(),'loss_IP':(loss_ip_B + loss_ip_A).cpu()}, 
                    images={'real_A': real_A.cpu(), 'real_B': real_B.cpu(), 'fake_A': fake_A.cpu(), 'fake_B': fake_B.cpu()})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2Bs.state_dict(), './cycGAN/netG_A2Bs.pth')
    torch.save(netG_B2As.state_dict(), './cycGAN/netG_B2As.pth')
    torch.save(netG_share.state_dict(), './cycGAN/netG_share.pth')
    torch.save(netD_A.state_dict(), './cycGAN/netD_A.pth')
    torch.save(netD_B.state_dict(), './cycGAN/netD_B.pth')