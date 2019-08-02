import torch
import torch.nn as nn
from torch.nn import init
from DL_torch.network import *
from DL_torch.util import *
import torch.nn.functional as F
import numpy as np

def define_GDA(input_cn, output_cn, which_model_netG, outputm_cn = 32, resnet_output_idx = None, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    if which_model_netG == 'DADenseCNNGenerator':
        netG = DADenseCNNGenerator(input_cn, output_cn, outputm_cn = 32, growth_cn = 8, nb_layers = 4, resnet_output_idx = resnet_output_idx)
    elif which_model_netG == 'DADenseCNNGenerator1':
        netG = DADenseCNNGenerator1(input_cn, output_cn, outputm_cn = 32, growth_cn = 8, nb_layers = 4, resnet_output_idx = resnet_output_idx)
    elif which_model_netG == 'DADenseCNNGenerator2':
        netG = DADenseCNNGenerator2(input_cn, output_cn, outputm_cn = 32, growth_cn = 8, nb_layers = 4, resnet_output_idx = resnet_output_idx)
    elif which_model_netG == 'DADenseCNNGenerator3':
        netG = DADenseCNNGenerator3(input_cn, output_cn, outputm_cn = 32, growth_cn = 8, nb_layers = 4, resnet_output_idx = resnet_output_idx)
    elif which_model_netG == 'DAUnetGenerator':
        netG = DAUnetGenerator(input_cn, output_cn, outputm_cn = 32, growth_cn = 8, nb_layers = 4, resnet_output_idx = resnet_output_idx)
    elif which_model_netG == 'DAUnetGenerator1':
        netG = DAUnetGenerator1(input_cn, output_cn, outputm_cn = 32, growth_cn = 8, nb_layers = 4, resnet_output_idx = resnet_output_idx)
    return init_net(netG, init_type, init_gain, gpu_ids)    


def define_G(input_cn, output_cn, ngf, which_model_netG, resnet_output_idx = None, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None

    if which_model_netG == 'ResUNet':
        netG = ResUNetGenerator(input_cn, output_cn, resnet_output_idx = resnet_output_idx, u_base_cn = ngf)
    elif which_model_netG == 'ResCNN':
        netG = ResCNNGenerator(input_cn, output_cn, resnet_output_idx = resnet_output_idx, u_base_cn = ngf)
    elif which_model_netG == 'RDCNN1':
        netG = ResDenseCNNGenerator1(input_cn, output_cn, growth_cn = 8, nb_layers = 6, resnet_output_idx = resnet_output_idx)
        #netG = UnetModule(input_nc, output_nc, base_cn = 8)
    elif which_model_netG == 'ResUNetModule':
        netG = ResUNetGenerator(input_cn, output_cn, resnet_output_idx = resnet_output_idx)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)

def define_G_sub(input_cn, output_cn, which_model_netG, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None

    if which_model_netG == 'CNNModule_share':
        netG = CNNModule(input_cn, output_cn, base_cn = 8, cnn_layers = 4, bn_layer = True, bn_layer_e = True)
    elif which_model_netG == 'CNNModule_split':
        netG = CNNModule(input_cn, output_cn, base_cn = 8, cnn_layers = 4,bn_layer = True, bn_layer_e = False)
    elif which_model_netG == 'ResNetModule':
        netG = ResNetModule(input_cn)
    
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)

def define_D(input_cn, ndf, which_model_netD, n_layers_D=3, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None

    if which_model_netD == 'basic':
        netD =  Discriminator(input_nc = input_cn, base_cn = 16,ndf = ndf, n_layers=n_layers_D)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)
