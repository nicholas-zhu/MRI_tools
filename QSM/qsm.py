import numpy as np
from FFT import fft
from util import *
from scipy import optimize

def Ball_filter(r, voxelsize):
    yy,xx,zz = np.meshgrid(np.arange(-r,r+1),np.arange(-r,r+1),np.arange(-r,r+1))
    xx = xx*voxelsize[0]
    yy = yy*voxelsize[1]
    zz = zz*voxelsize[2]
    v2 = xx**2 + yy**2 + zz**2 + np.spacing(1)
    B_kernal = v2<=(r**2+r/3)&v2>.1
    B_kernal = B_kernal/np.sum(B_kernal)
    return B_kernal

def SMVFiltering(Phase,radius, voxelsize = [1,1,1], pad_size = [6,6,6]):
    phase_pad = process_3d.zpad(Phase,pad_size)
    kernal = Ball_filter(radius,voxelsize)
    k_pad = process_3d.zpad2(kernal,phase_pad.shape[0:2])
    A = fft.fft(shape=1,axes=(0,1,2))
    if phase_pad.ndim>3 :
        kernal = np.expand_dims(kernal, axis=3)
    phase_L = A.IFT(A.FT(k_pad)*A.FT(phase_pad))
    phase_H = phase_pad - phase_L
    phase_L = process_3d.crop(phase_L,pad_size)
    phase_H = process_3d.crop(phase_H,pad_size)
    return phase_L, phase_H

def HARPERELLA(Phase,Mask,pad_size,voxel_size):
    
    yy, xx, zz = np.meshgrid(np.arange(0, raw_phase.shape[1]),
                             np.arange(0, raw_phase.shape[0]),
                             np.arange(0, raw_phase.shape[2]))
    xx, yy, zz = ((xx - np.round((raw_phase.shape[0])/2)) / field_of_view[0],
                  (yy - np.round((raw_phase.shape[1])/2)) / field_of_view[1],
                  (zz - np.round((raw_phase.shape[2])/2)) / field_of_view[2])
    k2 = xx**2 + yy**2 + zz**2 + np.spacing(1)
    k2 = np.square(xx) + np.square(yy) + np.square(zz)+ np.spacing(1)
    ik2 = 1/k2
    
    LP = A.IFT(k2*A.FT(Phase))
    Mask1 = SVMFiltering(Mask,3)
    MaskRim = 2*Mask - (Mask1>.99)
    b = A.IFT(ik2*A.FT(Mask*LP))*MaskRim
    
    Maskout = 1-Mask
    
    
    