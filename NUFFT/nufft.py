import numpy as np
import NUFFT
from FFT import fft
from NUFFT import kb128
from numba import jit

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
    def __init__(self, traj, os = 1.5, grid_r = None, pattern = None):
        self.traj = np.reshape(traj,[3,-1])*os
        self.samples = self.traj.shape[1]
        if grid_r is None:
            self.grid_r = (np.stack([np.min(self.traj,axis=1), np.max(self.traj,axis=1)],axis=1)).astype(np.int32)
            print(self.grid_r)
        else:
            self.grid_r = grid_r
        self.A = fft.fft(shape=1,axes=(0,1,2))
        self.p = np.reshape(pattern,[1,-1])
        
    def forward(self,img_c):
        data_c = self.A.FT(img_c)
        data_n = np.zeros([1,self.samples],dtype = np.complex128)
        data_n = grid(self.traj, data_c, data_n, self.grid_r)
        
        return data_n
        
    def adjoint(self,data_n):
        g = self.grid_r
        data_c = np.zeros([g[0,1]-g[0,0]+1,g[1,1]-g[1,0]+1,g[2,1]-g[2,0]+1],dtype = np.complex128)
        if self.p is None:
            data_c = gridH(self.traj, data_c, data_n, self.grid_r)
        else:
            data_c = gridH(self.traj, data_c, data_n*self.p, self.grid_r)
        img_hc = self.A.IFT(data_c)
        
        return img_hc
    
@jit(nopython = True)
def KB_3d(dis, width_scale, ndim = 3):
    # ndim = 3, dis:3x1
    # w griding weighting
    w = 1
    scaled_grid = np.abs(dis)*width_scale
    for i in range(ndim):
        w1 = scaled_grid[i] - np.floor(scaled_grid[i])
        kb_1 = np.floor(scaled_grid[i])
        w = w * (w1*kb_table[kb_1] + (1-w1) *kb_table[kb_1+1])

    return w

@jit(nopython = True)
def grid(traj, data_c, data_n, grid_r, dis, width = 3.5):
    # weight calculation
    samples = traj.shape[1]
    width_scale = (KB_table.size-1)/width
    print(samples.shape)
    # data_n = np.zeros(np.array([1,samples],dtype = np.int32),dtype = np.complex128)
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:] 
    
    nx = (width*2).astype(int32)
    ny = (width*2).astype(int32)
    nz = (width*2).astype(int32)
    
    for n in range(samples):
        ind_x = np.arange(np.ceil(np.maximum(kx[n]-width,grid_r[0,0])),np.floor(np.minimum(kx[n]+width,grid_r[0,1])))
        ind_y = np.arange(np.ceil(np.maximum(ky[n]-width,grid_r[1,0])),np.floor(np.minimum(ky[n]+width,grid_r[1,1])))
        ind_z = np.arange(np.ceil(np.maximum(kz[n]-width,grid_r[2,0])),np.floor(np.minimum(kz[n]+width,grid_r[2,1])))
        for i in ind_x:
            dis[0] = np.abs(ind_x - kx[n])
            cx = (i-grid_r[0,0]).astype(np.int32)
            for j in ind_y:
                dis[1] = np.abs(ind_y - ky[n])
                cy = (j-grid_r[1,0]).astype(np.int32)
                for k in ind_z:
                    dis[2] = np.abs(ind_z - kz[n])
                    cz = (k-grid_r[2,0]).astype(np.int32)
                    w = KB_3d(dis,width_scale)
                    data_n[0,n] = data_n[0,n] + data_c[cx,cy,cz] * w
                    
    return data_n
                
@jit(nopython = True)
def gridH(traj, data_c, data_n, grid_r, dis, width = 3.5):
    # weight calculation
    samples = traj.shape[1]
    width_scale = (KB_table.size-1)/width
    print(samples.shape)
    # data_n = np.zeros(np.array([1,samples],dtype = np.int32),dtype = np.complex128)
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:] 
    
    nx = (width*2).astype(int32)
    ny = (width*2).astype(int32)
    nz = (width*2).astype(int32)
    
    for n in range(samples):
        ind_x = np.arange(np.ceil(np.maximum(kx[n]-width,grid_r[0,0])),np.floor(np.minimum(kx[n]+width,grid_r[0,1])))
        ind_y = np.arange(np.ceil(np.maximum(ky[n]-width,grid_r[1,0])),np.floor(np.minimum(ky[n]+width,grid_r[1,1])))
        ind_z = np.arange(np.ceil(np.maximum(kz[n]-width,grid_r[2,0])),np.floor(np.minimum(kz[n]+width,grid_r[2,1])))
        for i in ind_x:
            dis[0] = np.abs(ind_x - kx[n])
            cx = (i-grid_r[0,0]).astype(np.int32)
            for j in ind_y:
                dis[1] = np.abs(ind_y - ky[n])
                cy = (j-grid_r[1,0]).astype(np.int32)
                for k in ind_z:
                    dis[2] = np.abs(ind_z - kz[n])
                    cz = (k-grid_r[2,0]).astype(np.int32)
                    w = KB_3d(dis,width_scale)
                    data_c[cx,cy,cz] = data_c[cx,cy,cz] + data_n[0,n] * w
        if samples%100000==0:
            print(n)
                    
    return data_c