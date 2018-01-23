import numpy as np
import NUFFT
from FFT import fft
from NUFFT import kb128

# grid, gridH operators
# TODO: precompute
#
# input:
#     traj   : non-cartesian traj
#     grid   : cartesian grid
#     data_c : cartesian data
# output:
#     data_n : non_cartesian data

kb_table = kb128.kb128

class NUFFT():
    def __init__(self, traj, grid_r, pattern = None):
        self.traj = traj
        self.grid_r = grid_r
        self.A = fft.fft(shape=1,axes=(0,1,2))
        
    def forward(img_c):
        data_c = self.A.FT(img_c)
        data_n = grid(self.traj, data_c, self.grid_r)
        
        return data_n
        
    def adjoint(data_n):
        data_c = gridH(self.traj, data_n, self.grid_r)
        img_hc = self.A.IFT(data_c)
        
        return img_hc

def KB_3d(grid, kb_table, width):
    # grid[N,3] kb_table[128]
    # low accuracy
    scaled_grid = np.floor(np.abs(grid)/(2*width)*(kb_table.size-1))
    wx = kb_table[scaled_grid[:,0].astype(int)]
    wy = kb_table[scaled_grid[:,1].astype(int)]
    wz = kb_table[scaled_grid[:,2].astype(int)]
    
    return wx*wy*wz

def grid(traj, data_c, grid_r, width = 3, oversample = 1.5):
    # weight calculation
    # grid_data = np.zeros([grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]])
    samples = traj.shape[1]
    data_n = np.zeros([0,samples])
    kx = oversample * traj[0,:]
    ky = oversample * traj[1,:]
    kz = oversample * traj[2,:] 
    for i in range(samples):
        ind_x = np.arange(np.round(np.maximum(kx[i]-width,grid_r[0,0])),np.floor(np.minimum(kx[i]+width,grid_r[0,1])))
        ind_y = np.arange(np.round(np.maximum(ky[i]-width,grid_r[1,0])),np.floor(np.minimum(ky[i]+width,grid_r[1,1])))
        ind_z = np.arange(np.round(np.maximum(kz[i]-width,grid_r[2,0])),np.floor(np.minimum(kz[i]+width,grid_r[2,1])))
        kgrid_y,kgrid_x,kgrid_z = np.meshgrid(ind_y,ind_x,ind_z)
        kgrid = np.stack((kgrid_x.flatten()-kx[i],kgrid_y.flatten()-ky[i],kgrid_z.flatten()-kz[i]),axis=1)
        weight = KB_3d(kgrid,kb_table,width)
        kernel = data_c[kgrid_x.reshape(-1).astype(int),kgrid_y.reshape(-1).astype(int),kgrid_z.reshape(-1).astype(int)]
        data_n[0,i] = np.sum(kernel*weight)
        
    return data_n

def gridH(traj, data_n, grid_r, width = 3, oversample = 1.5):
    
    samples = traj.shape[1]
    data_c = np.zeros([grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]])
    kx = oversample * traj[0,:]
    ky = oversample * traj[1,:]
    kz = oversample * traj[2,:] 
    for i in range(samples):
        ind_x = np.arange(np.round(np.maximum(kx[i]-width,grid_r[0,0])),np.floor(np.minimum(kx[i]+width,grid_r[0,1])))
        ind_y = np.arange(np.round(np.maximum(ky[i]-width,grid_r[1,0])),np.floor(np.minimum(ky[i]+width,grid_r[1,1])))
        ind_z = np.arange(np.round(np.maximum(kz[i]-width,grid_r[2,0])),np.floor(np.minimum(kz[i]+width,grid_r[2,1])))
        kgrid_y,kgrid_x,kgrid_z = np.meshgrid(ind_y,ind_x,ind_z)
        kgrid = np.stack((kgrid_x.flatten()-kx[i],kgrid_y.flatten()-ky[i],kgrid_z.flatten()-kz[i]),axis=1)
        weight = KB_3d(kgrid,kb_table,width)
        kernel = weight*data_n[0,i]
        data_c[kgrid_x.reshape(-1).astype(int),kgrid_y.reshape(-1).astype(int),kgrid_z.reshape(-1).astype(int)] += kernel
        
    return data_c
    
    