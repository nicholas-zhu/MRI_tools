import numpy as np
import tensorflow as tf
from FFT import fft
import matplotlib.pyplot as plt

def MRI_mask_2d(Img_shape, sampling_rate, full_area, N):
    # Mask [N, nx, ny]
    fx = np.minimum(full_area[0],Img_shape[0])
    fy = np.minimum(full_area[1],Img_shape[1])
    # how to transfer to poisson disc
    Mask = (np.random.rand(N,Img_shape[0],Img_shape[1])>(1-sampling_rate))
    
    Mask[:,(Img_shape[0]-fx)//2:(fx-Img_shape[0])//2,(Img_shape[0]-fx)//2:(fx-Img_shape[0])//2] = 1

    return Mask

def MRI_Phase_2d(Img_shape, N):
    Lx = np.arange(Img_shape[0])/Img_shape[0]
    Ly = np.arange(Img_shape[1])/Img_shape[1]
    grid_y,grid_x = np.meshgrid(Ly,Lx)
    grid_x = grid_x[None,:,:]
    grid_y = grid_y[None,:,:]
    
    c2 = np.random.randn(N)[:,None,None]*2
    c2xy = np.random.randn(N)[:,None,None]*2
    c1x = np.random.randn(N)[:,None,None]*4
    c1y = np.random.randn(N)[:,None,None]*4
    c0 = np.random.randn(N)[:,None,None]
    
    phase = c2*(grid_x**2-grid_y**2) + c2xy*grid_x*grid_y + c1x*grid_x + c1y*grid_y +c0
    
    return phase

def MRI_Noise_2d(Img_shape, N, sigma = 1):
    Noise_i = np.random.randn(N,Img_shape[0],Img_shape[1])*sigma
    Noise_r = np.random.randn(N,Img_shape[0],Img_shape[1])*sigma
    
    return 1j * Noise_i + Noise_r
    
    
def MRI_undersample_2d(Img, sampling_rate = .7, full_area = [24,24], N = 10, sigma = 1):
    # Img [nx, ny]
    # random image
    # random phase
    Img_shape = Img.shape
    phase = MRI_Phase_2d(Img_shape,N)
    Img_true = Img[None,:,:]*np.exp(1j*phase)
    FFT = fft.fft(shape=1,axes=(1,2))
    # FFT
    ksp = FFT.FT(Img_true)

    

    sigma = sigma * np.max(np.abs(ksp.flatten()))
    noise = MRI_Noise_2d(Img_shape,N,sigma = sigma)
    Mask = MRI_mask_2d(Img_shape, sampling_rate, full_area, N)
    Img_out = FFT.IFT((ksp+noise)*Mask)
    
    return Img_true, Img_out