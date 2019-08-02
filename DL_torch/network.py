import torch
import torch.nn as nn
from torch.nn import init
from DL_torch.DL_cell import *
import torch.nn.functional as F
from scipy import signal

__all__ = ['init_net','CNNModule','ResNetModule','ResCNNGenerator','DAUnetGenerator','DAUnetGenerator1','ResDenseCNNGenerator1','DADenseCNNGenerator','DADenseCNNGenerator1','DADenseCNNGenerator2','DADenseCNNGenerator3','ResUNetGenerator','Discriminator', 'SSIM_loss','SSIM_loss1']

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

class Discriminator(nn.Module):
    """ Defines a PatchGAN discriminator
        Borrow from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
    """
    def __init__(self, input_nc, base_cn = 8, ndf=64, n_layers=3):
        """ Construct a PatchGAN discriminator
            Parameters:
                input_nc (int)  -- the number of channels in input images
                ndf (int)       -- the number of filters in the last conv layer
                n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(Discriminator, self).__init__()

        base_cn_s = 2
        sequence = []
        sequence.append(Conv2d_block(input_nc, base_cn))
        
        for n in range(n_layers):
            sequence.append(Maxpool2d_block(base_cn*base_cn_s**n, base_cn*base_cn_s**(n+1)))            
            # sequence.append(Conv2d_down(base_cn*base_cn_s**n, base_cn*base_cn_s**(n+1)))
            sequence.append(Conv2d_block(base_cn*base_cn_s**(n+1), base_cn*base_cn_s**(n+1)))
        base_cn = base_cn*(base_cn_s**n_layers)

        sequence.append(Conv2d_block( base_cn, 1024, kernel = 1, drop_out = 0.5, afunc = torch.relu))
        sequence.append(Conv2d_block( 1024, 1,kernel = 1, drop_out = 0.5, afunc = torch.sigmoid))
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        x = self.model(input)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return x

######## loss function ########  
def _create_window(window_size, channel):
        _1D_window = torch.from_numpy(np.ones(window_size)/window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    
def _ssim(img1, img2, window, window_size, channel, size_average = True, mask = None, non_neg_flag = False):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
        L = 128 # how to tune this?
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        # window2 = window ** 2
        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = (L*0.01)**2
        C2 = (L*0.03)**2
        
        ret = (2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
        cs = (2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
        if non_neg_flag is False:
            ssim_map = ret * cs 
        else:
            ssim_map = (ret + 1)/2 * (cs + 1)/2
        
        if mask is not None:
            ssim_map = ssim_map * mask / (mask.mean()+1e-6)
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
        
class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = _create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
class SSIM_loss1(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True, non_neg_flag = False):
        super(SSIM_loss1, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = _create_window(window_size, self.channel)
        self.non_neg_flag = non_neg_flag
        
    def forward(self, img1, img2, mask = None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask, self.non_neg_flag)    
        
    
    
    
######## basic network structure ########
class CNNModule(nn.Module):
    def __init__(self, input_cn, output_cn, base_cn = 8, cnn_layers = 3, bn_layer = True, bn_layer_e = False):
        """ Construct a CNN Module
        Parameters:
        """
        super(CNNModule, self).__init__()
        
        self.cnn_layers = cnn_layers
        if self.cnn_layers <= 1:
            base_cn = output_cn
        sequence = [Conv2d_block(input_cn, base_cn)]
        for i in range(self.cnn_layers-1):
            sequence.append(Conv2d_block(base_cn,base_cn,bn_layer = bn_layer))
        if bn_layer_e:
            sequence.append(Conv2d_block(base_cn, output_cn, bn_layer = bn_layer_e))
        else:
            sequence.append(Conv2d_block(base_cn, output_cn, afunc = None, bn_layer = bn_layer_e))
        self.model = nn.Sequential(*sequence)
        
    def forward(self, input): 
        return self.model(input)


class UnetModule(nn.Module):
    def __init__(self, input_cn, output_cn, base_cn = 8, unet_level = 3, up_conv_flag = True):
        """ Construct a UNet module
        Parameters:
        """
        super(UnetModule, self).__init__()
        base_cn_s = 2
        
        self.unet_level = unet_level
        self.conv0 = Conv2d_out(input_cn, base_cn)
        self.conv = nn.ModuleList([])
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        self.conv_u = nn.ModuleList([])
        self.concat = nn.ModuleList([])
        
        base_cnt = base_cn
        self.conv.append(Conv2d_block(base_cnt,base_cnt))
        for i in range(unet_level):
            self.down.append(Conv2d_down(base_cnt, base_cnt*base_cn_s))
            self.conv.append(Conv2d_block(base_cnt*base_cn_s,base_cnt*base_cn_s))
            base_cnt = base_cn * base_cn_s**(i+1)
        
        for j in range(unet_level):
            base_cnt = base_cn * base_cn_s**(unet_level-j)
            if up_conv_flag:
                self.up.append(Conv2d_up(base_cnt, base_cnt//base_cn_s))
            else:
                self.up.append(nn.Sequential(*nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), Conv2d_block(base_cnt,base_cnt//base_cn_s)])))
            self.conv_u.append(Conv2d_block(base_cnt,base_cnt//base_cn_s))
            self.concat.append(Concat2d())
            
        # self.out = Conv2d_out(base_cn, output_cn)
        self.out = Conv2d_block(base_cn, output_cn)
        
    def forward(self, input):
        
        x0 = self.conv0(input)
        x = []
        x.append(self.conv[0](x0))
        # going down
        for i in range(self.unet_level):
            xt = x[-1]
            xt = self.down[i](xt)
            x.append(self.conv[i+1](xt))
        
        xt = x[self.unet_level]
        for i in range(self.unet_level):
            xt = self.up[i](xt)
            xt = self.concat[i](x[self.unet_level-i-1],xt)
            xt = self.conv_u[i](xt)
            
        return self.out(xt)    
    
class DenseBlock(nn.Module):
    def __init__(self, nb_layers, input_cn, growth_cn, output_cn, block, dropRate=0.5):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, input_cn, output_cn, growth_cn, nb_layers, dropRate)
    def _make_layer(self, block, input_cn, output_cn, growth_cn, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(input_cn+i*growth_cn, growth_cn, drop_out = dropRate))
        layers.append(Conv2d_out(input_cn+nb_layers*growth_cn, output_cn))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
    
class ResNetModule(nn.Module):
    def __init__(self, input_cn, output_idx = None):
        super(ResNetModule, self).__init__()
        self.input_cn = input_cn
        self.output_idx = output_idx
        
    def forward(self,input, NN_block):
        
        if self.output_idx is not None:
            output = input[:,self.output_idx:self.output_idx+1,...]
        else:
            output = input
            
        return output + NN_block(input)

class ResNet2Module(nn.Module):
    def __init__(self, input_cn, output_idx = None):
        super(ResNet2Module, self).__init__()
        self.input_cn = input_cn
        self.output_idx = output_idx
        
    def forward(self,input, resinput):
        
        if self.output_idx is not None:
            output = input[:,self.output_idx:self.output_idx+1,...]
        else:
            output = input
            
        return output + resinput    
    
class ResDenseCNNGenerator1(nn.Module):
    def __init__(self,  input_cn, output_cn, growth_cn = 8, nb_layers = 4,resnet_output_idx = None, dropRate=0.5):
        super(ResDenseCNNGenerator1, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.RDCNN = DenseBlock(nb_layers, input_cn, growth_cn, output_cn, ConvCat2d_block, dropRate=0.5)
        self.ResNet = ResNetModule(output_cn,self.resnet_output_idx)
        
    def forward(self,input):
        output = self.ResNet(input, self.RDCNN)
        return output
    
class DADenseCNNGenerator(nn.Module):
    def __init__(self,  input_cn, output_cn, outputm_cn = 32, growth_cn = 8, nb_layers = 4,resnet_output_idx = None, dropRate=0.5):
        super(DADenseCNNGenerator, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.Dense1 = DenseBlock(nb_layers, input_cn, growth_cn, outputm_cn, ConvCat2d_block, dropRate=0.5)
        self.Dense2 = DenseBlock(nb_layers, input_cn, growth_cn, outputm_cn, ConvCat2d_block, dropRate=0.5)
        self.Dense3 = DenseBlock(nb_layers, outputm_cn, growth_cn, output_cn, ConvCat2d_block, dropRate=0.5)
        self.ActScale = nn.Sigmoid()
        self.ResNet = ResNet2Module(output_cn,self.resnet_output_idx)
        
    def forward(self,input, input_A):
        x = self.Dense1(input)*self.ActScale(self.Dense2(input_A))
        x = self.Dense3(x)
        output = self.ResNet(input, x)
        return output

class DAUnetGenerator1(nn.Module):
    def __init__(self,  input_cn, output_cn, outputm_cn = 64, growth_cn = 8, nb_layers = 4,resnet_output_idx = None, dropRate=0.5):
        super(DAUnetGenerator1, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.UNet1 = DenseBlock(nb_layers, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.UNet2_1 = DenseBlock(2, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.UNet2_2 = UnetModule(input_cn, outputm_cn, base_cn = growth_cn , unet_level = nb_layers)
        self.UNet3 = DenseBlock(2, outputm_cn, growth_cn, output_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.ActScale = nn.Softmax2d()
        self.ResNet = ResNet2Module(output_cn,self.resnet_output_idx)
        self.out = nn.ReLU()
    def forward(self,input, input_A):
        # TODO visualize attention
        amap = torch.sigmoid(self.UNet2_2(input))*self.ActScale(self.UNet2_1(input_A))
        x = self.UNet1(input) * amap
        x = self.UNet3(x)
        output = self.out(self.ResNet(input, x))
        return output, amap      
    
class DAUnetGenerator(nn.Module):
    def __init__(self,  input_cn, output_cn, outputm_cn = 64, growth_cn = 8, nb_layers = 4,resnet_output_idx = None, dropRate=0.5):
        super(DAUnetGenerator, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.UNet1 = UnetModule(input_cn, outputm_cn, base_cn = growth_cn , unet_level = nb_layers)
        self.UNet2_1 = DenseBlock(2, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.UNet2_2 = DenseBlock(2, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.UNet3 = DenseBlock(2, outputm_cn, growth_cn, output_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.ActScale = nn.Softmax2d()
        self.ResNet = ResNet2Module(output_cn,self.resnet_output_idx)
        self.out = nn.ReLU()
    def forward(self,input, input_A):
        # TODO visualize attention
        amap = torch.sigmoid(self.UNet2_2(input))*self.ActScale(self.UNet2_1(input_A))
        x = self.UNet1(input) * amap
        x = self.UNet3(x)
        output = self.out(self.ResNet(input, x))
        return output, amap     
    
class DADenseCNNGenerator3(nn.Module):
    def __init__(self,  input_cn, output_cn, outputm_cn = 64, growth_cn = 8, nb_layers = 4,resnet_output_idx = None, dropRate=0.5):
        super(DADenseCNNGenerator3, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.Dense1 = DenseBlock(nb_layers, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.Dense2_1 = DenseBlock(nb_layers, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.Dense2_2 = DenseBlock(nb_layers, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.Dense3 = DenseBlock(nb_layers, outputm_cn, growth_cn, output_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.ActScale = nn.Softmax2d()
        self.ResNet = ResNet2Module(output_cn,self.resnet_output_idx)
        self.out = nn.ReLU()
    def forward(self,input, input_A):
        # TODO visualize attention
        amap = torch.sigmoid(self.Dense2_2(input))*self.ActScale(self.Dense2_1(input_A))
        x = self.Dense1(input) * amap
        x = self.Dense3(x)
        output = self.out(self.ResNet(input, x))
        return output, amap     
    
class DADenseCNNGenerator2(nn.Module):
    def __init__(self,  input_cn, output_cn, outputm_cn = 64, growth_cn = 8, nb_layers = 4,resnet_output_idx = None, dropRate=0.5):
        super(DADenseCNNGenerator2, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.Dense1 = DenseBlock(nb_layers, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.Dense2_1 = DenseBlock(2, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.Dense2_2 = DenseBlock(2, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.Dense3 = DenseBlock(nb_layers, outputm_cn, growth_cn, output_cn, ConvCat2d_block, dropRate=0.5)
        self.ActScale = nn.Softmax2d()
        self.ResNet = ResNet2Module(output_cn,self.resnet_output_idx)
        
    def forward(self,input, input_A):
        x = self.Dense1(input)*self.ActScale(self.Dense2_2(input)*self.Dense2_1(input_A))
        x = self.Dense3(x)
        output = self.ResNet(input, x)
        return output    
    
class DADenseCNNGenerator1(nn.Module):
    def __init__(self,  input_cn, output_cn, outputm_cn = 32, growth_cn = 8, nb_layers = 4,resnet_output_idx = None, dropRate=0.5):
        super(DADenseCNNGenerator1, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.Dense1 = DenseBlock(nb_layers, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.conv1 = nn.Conv2d(input_cn, outputm_cn, 3, padding = 1)
        self.Dense2 = DenseBlock(2, input_cn, growth_cn, outputm_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.Dense3 = DenseBlock(nb_layers, outputm_cn, growth_cn, output_cn, prea_ConvCat2d_block, dropRate=0.5)
        self.ActScale = nn.Softmax2d()
        self.ResNet = ResNet2Module(output_cn,self.resnet_output_idx)
        
    def forward(self,input, input_A):
        x = self.Dense1(input)*self.ActScale(self.conv1(input)*self.Dense2(input_A))
        x = self.Dense3(x)
        output = self.ResNet(input, x)
        return output
    
    
class ResCNNGenerator(nn.Module):
    def __init__(self, input_cn, output_cn, resnet_output_idx = None, u_base_cn = 16 ):
        super(ResCNNGenerator, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.CNNNet = CNNModule(input_cn, output_cn, base_cn = u_base_cn)
        self.ResNet = ResNetModule(output_cn,self.resnet_output_idx)
        
    def forward(self,input):
        output = self.ResNet(input, self.CNNNet)
        return output    
    
class ResUNetGenerator(nn.Module):
    def __init__(self, input_cn, output_cn, resnet_output_idx = None, u_base_cn = 8 ):
        super(ResUNetGenerator, self).__init__()
        if resnet_output_idx is None:
            assert input_cn == output_cn, "Generator Input/Output channels mismatch!"
        else:
            assert output_cn == 1, "Generator num of Output channel is expected to be 1!"
        self.input_cn = input_cn
        self.output_cn = output_cn
        self.resnet_output_idx = resnet_output_idx
        self.UNet = UnetModule(input_cn, output_cn, base_cn = u_base_cn)
        self.ResNet = ResNetModule(output_cn,self.resnet_output_idx)
        self.out = nn.ReLU()
        
    def forward(self,input):
        output = self.out(self.ResNet(input, self.UNet))
        
        return output
