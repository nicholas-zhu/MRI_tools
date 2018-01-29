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

kb_table2 = kb128.kb128

class NUFFT():
    def __init__(self, traj, grid_r = None, os = 1.2, pattern = None, width = 3):
        self.traj = os*np.reshape(traj,[3,-1])
        self.samples = (self.traj).shape[1]
        if grid_r is None:
            self.grid_r = (np.stack([np.floor(np.min(self.traj,axis=1))-width, np.ceil(np.max(self.traj,axis=1))+width],axis=1)).astype(np.int32)
            print(self.grid_r)
        else:
            self.grid_r = grid_r.astype(np.int32)
        if pattern is None:
            self.p = None
        else:
            self.p = np.reshape(pattern,[1,-1])
        self.A = fft.fft(shape=1,axes=(0,1,2))
        self.width = width
        self.win = self.A.IFT(KB_compensation(self.grid_r,width))
        # kb psf compensation
        
    def forward(self,img_c):
        data_c = self.A.FT(img_c)
        data_n = grid(self.samples,self.traj, data_c, self.grid_r, width =self.width)
        
        return data_n
        
    def adjoint(self,data_n):
        data_n = np.reshape(data_n,[1,-1])
        if self.p is None:
            data_c = gridH2(self.samples,self.traj, data_n, self.grid_r, width =self.width)
        else:
            data_c = gridH2(self.samples,self.traj, data_n*self.p, self.grid_r, width =self.width)
        img_hc = self.A.IFT(data_c)
        
        return img_hc
    
    def ATA(self,img_c):
        data_c = self.A.FT(img_c)
        data_ct = gTg(self.samples,self.traj, data_c, self.grid_r, width =self.width)
        img_ct = self.A.IFT(data_ct)
        
        return img_ct
    
    
def KB_compensation(grid_r, width):
    win = np.zeros([grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]],dtype = np.complex128)
    c_ind = -grid_r[:,0]
    win_ind = (np.arange(np.ceil(-width),np.floor(width))).astype(np.int32)
    kx,ky,kz = np.meshgrid(win_ind,win_ind,win_ind)
    kgrid = np.stack((kx.flatten(),ky.flatten(),kz.flatten()),axis=1)
    win[kgrid[:,0]+c_ind[0],kgrid[:,1]+c_ind[1],kgrid[:,2]+c_ind[2]] = KB_3d(kgrid,kb_table2,width)
    return win
        

def KB_3d(grid, kb_table, width):
    # grid[N,3] kb_table[128]
    # low accuracy
    scaled_grid = np.floor(np.abs(grid)/(width)*(kb_table.size-1)).astype(np.int32)
    wx = kb_table[scaled_grid[:,0]]
    wy = kb_table[scaled_grid[:,1]]
    wz = kb_table[scaled_grid[:,2]]
    
    return wx*wy*wz

def grid(samples, traj, data_c, grid_r, width):
    # weight calculation
    kb_table = kb_table2
    data_n = np.zeros(np.array([1,samples]),dtype = np.complex128)
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:] 
    for i in range(samples):
        ind_x = np.arange(np.ceil(np.maximum(kx[i]-width,grid_r[0,0])),np.floor(np.minimum(kx[i]+width,grid_r[0,1]-1)))
        ind_y = np.arange(np.ceil(np.maximum(ky[i]-width,grid_r[1,0])),np.floor(np.minimum(ky[i]+width,grid_r[1,1]-1)))
        ind_z = np.arange(np.ceil(np.maximum(kz[i]-width,grid_r[2,0])),np.floor(np.minimum(kz[i]+width,grid_r[2,1]-1)))
        kgrid_y,kgrid_x,kgrid_z = np.meshgrid(ind_y,ind_x,ind_z)
        kgrid = np.stack((kgrid_x.flatten()-kx[i],kgrid_y.flatten()-ky[i],kgrid_z.flatten()-kz[i]),axis=1)
        weight = KB_3d(kgrid,kb_table,width)
        kernel = data_c[(kgrid_x.reshape(-1)-grid_r[0,0]).astype(int),(kgrid_y.reshape(-1)-grid_r[1,0]).astype(int),(kgrid_z.reshape(-1)-grid_r[2,0]).astype(int)]
        data_n[0,i] = np.sum(kernel*weight)
        
    return data_n


def gridH(samples, traj, data_n, grid_r, width):
    
    kb_table = kb_table2
    data_c = np.zeros(np.array([grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]],dtype = np.int32),dtype = np.complex128)
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:] 
    for i in range(samples):
        ind_x = np.arange(np.ceil(np.maximum(kx[i]-width,grid_r[0,0])),np.floor(np.minimum(kx[i]+width,grid_r[0,1]-1)))
        ind_y = np.arange(np.ceil(np.maximum(ky[i]-width,grid_r[1,0])),np.floor(np.minimum(ky[i]+width,grid_r[1,1]-1)))
        ind_z = np.arange(np.ceil(np.maximum(kz[i]-width,grid_r[2,0])),np.floor(np.minimum(kz[i]+width,grid_r[2,1]-1)))
        kgrid_x,kgrid_y,kgrid_z = np.meshgrid(ind_x,ind_y,ind_z)
        kgrid = np.stack((kgrid_x.flatten()-kx[i],kgrid_y.flatten()-ky[i],kgrid_z.flatten()-kz[i]),axis=1)
        weight = KB_3d(kgrid,kb_table,width)
        kernel = weight*data_n[0,i]
        data_c[(kgrid_x.reshape(-1)-grid_r[0,0]).astype(int),(kgrid_y.reshape(-1)-grid_r[1,0]).astype(int),(kgrid_z.reshape(-1)-grid_r[2,0]).astype(int)] += kernel
        if i % 100000 ==0:
            print(i)
        
    return data_c
    
    
def gTg(samples, traj, data_c, grid_r, width):
    kb_table = kb_table2
    # weight calculation
    data_ct = np.zeros([grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]],dtype = np.complex128)
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:] 
    print(data_ct.shape)
    for i in range(samples):
        
        rind_x = np.arange(np.ceil(np.maximum(kx[i]-width,grid_r[0,0])),np.floor(np.minimum(kx[i]+width,grid_r[0,1]-1)))
        rind_y = np.arange(np.ceil(np.maximum(ky[i]-width,grid_r[1,0])),np.floor(np.minimum(ky[i]+width,grid_r[1,1]-1)))
        rind_z = np.arange(np.ceil(np.maximum(kz[i]-width,grid_r[2,0])),np.floor(np.minimum(kz[i]+width,grid_r[2,1]-1)))
        
        kgrid_x,kgrid_y,kgrid_z = np.meshgrid(rind_x,rind_y,rind_z)
        kgrid = np.stack((kgrid_x.flatten()-kx[i],kgrid_y.flatten()-ky[i],kgrid_z.flatten()-kz[i]),axis=1)
        
        aind_x = (kgrid_x.reshape(-1)-grid_r[0,0]).astype(int)
        aind_y = (kgrid_y.reshape(-1)-grid_r[1,0]).astype(int)
        aind_z = (kgrid_z.reshape(-1)-grid_r[2,0]).astype(int)
        
        weight = KB_3d(kgrid,kb_table2,width)        
        
        kernel = weight*np.inner(weight,data_c[aind_x,aind_y,aind_z])
        data_ct[aind_x,aind_y,aind_z] += kernel
        
        if i % 100000 ==0:
            print(i)
            
    return data_ct

def KB_3d2(grid_x, grid_y, grid_z, kb_table, width):
    # grid*[N,2*width] kb_table[128]
    # low accuracy
    scale = (width)/(kb_table.size-1)
    frac_x = np.minimum(np.abs(grid_x)/scale,kb_table.size-2)
    frac_y = np.minimum(np.abs(grid_y)/scale,kb_table.size-2)
    frac_z = np.minimum(np.abs(grid_z)/scale,kb_table.size-2)
    grid_xs = np.floor(frac_x).astype(np.int32)
    grid_ys = np.floor(frac_y).astype(np.int32)
    grid_zs = np.floor(frac_z).astype(np.int32)
    frac_x = frac_x-grid_xs
    frac_y = frac_y-grid_ys
    frac_z = frac_z-grid_zs
    
    wx = (1-frac_x)*kb_table[grid_xs] + frac_x*kb_table[grid_xs+1]
    wy = (1-frac_y)*kb_table[grid_ys] + frac_y*kb_table[grid_ys+1]
    wz = (1-frac_z)*kb_table[grid_zs] + frac_z*kb_table[grid_zs+1]
    
    return wx, wy, wz
                   

def gridH2(samples, traj, data_n, grid_r, width):
    # weight calculation
    kb_table = kb_table2
    data_c = np.zeros([grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]],dtype = np.complex128)
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:] 
    batch_size = 500000

    kernal_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    # test
    #for i in [1]:
    for i in range(samples//batch_size):
        print(i)
        batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
        rind_x = np.round(np.tile(kx[batch_ind][:,None],(1,k_len))+kernal_ind).astype(np.int32)
        rind_y = np.round(np.tile(ky[batch_ind][:,None],(1,k_len))+kernal_ind).astype(np.int32)
        rind_z = np.round(np.tile(kz[batch_ind][:,None],(1,k_len))+kernal_ind).astype(np.int32)
        kgrid_x = np.tile(rind_x[:,:,None,None],(1,1,k_len,k_len)).reshape([batch_ind.size,-1])
        kgrid_y = np.tile(rind_y[:,None,:,None],(1,k_len,1,k_len)).reshape([batch_ind.size,-1])
        kgrid_z = np.tile(rind_z[:,None,None,:],(1,k_len,k_len,1)).reshape([batch_ind.size,-1])
        
        wx,wy,wz = KB_3d2(np.abs(rind_x-kx[batch_ind][:,None]),np.abs(rind_y-ky[batch_ind][:,None]),np.abs(rind_z-kz[batch_ind][:,None]),kb_table,width)
        
        wx = np.tile(wx[:,:,None,None],(1,1,k_len,k_len)).reshape([batch_ind.size,-1])
        wy = np.tile(wy[:,None,:,None],(1,k_len,1,k_len)).reshape([batch_ind.size,-1])
        wz = np.tile(wz[:,None,None,:],(1,k_len,k_len,1)).reshape([batch_ind.size,-1])
        w = wx*wy*wz

        aind_x = (np.minimum(np.maximum(kgrid_x,grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
        aind_y = (np.minimum(np.maximum(kgrid_y,grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
        aind_z = (np.minimum(np.maximum(kgrid_z,grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])
        w_mask = (aind_x == kgrid_x-grid_r[0,0])*(aind_y == kgrid_y-grid_r[1,0])*(aind_z == kgrid_z-grid_r[2,0])
        w = w*w_mask
        
        kn = 0
        for k in batch_ind:
            w_k = w[kn,:]
            data_c[aind_x[kn],aind_y[kn],aind_z[kn]] += w_k*data_n[0,k]
            kn += 1
            
    return data_c    
    
def gTg2(samples, traj, data_c, grid_r, width):
    # weight calculation
    kb_table = kb_table2
    data_ct = np.zeros([grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]],dtype = np.complex128)
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:] 
    print(data_ct.shape)
    batch_size = 50000

    k_ind = np.arange(np.floor(-width),np.floor(width))
    k_ind = k_ind[None,:]
    k_len = k_ind.size
    for i in range(samples//batch_size):
        print(i)
        batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
        rind_x = np.round(np.tile(kx[batch_ind][:,None],(1,k_len))+kernal_ind).astype(np.int32)
        rind_y = np.round(np.tile(ky[batch_ind][:,None],(1,k_len))+kernal_ind).astype(np.int32)
        rind_z = np.round(np.tile(kz[batch_ind][:,None],(1,k_len))+kernal_ind).astype(np.int32)
        kgrid_x = np.tile(rind_x[:,:,None,None],(1,1,k_len,k_len)).reshape([batch_ind.size,-1])
        kgrid_y = np.tile(rind_y[:,None,:,None],(1,k_len,1,k_len)).reshape([batch_ind.size,-1])
        kgrid_z = np.tile(rind_z[:,None,None,:],(1,k_len,k_len,1)).reshape([batch_ind.size,-1])
        
        wx,wy,wz = KB_3d2(np.abs(rind_x-kx[batch_ind][:,None]),np.abs(rind_y-ky[batch_ind][:,None]),np.abs(rind_z-kz[batch_ind][:,None]),kb_table,width)
        
        wx = np.tile(wx[:,:,None,None],(1,1,k_len,k_len)).reshape([batch_ind.size,-1])
        wy = np.tile(wy[:,None,:,None],(1,k_len,1,k_len)).reshape([batch_ind.size,-1])
        wz = np.tile(wz[:,None,None,:],(1,k_len,k_len,1)).reshape([batch_ind.size,-1])
        w = wx*wy*wz

        aind_x = (np.minimum(np.maximum(kgrid_x,grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
        aind_y = (np.minimum(np.maximum(kgrid_y,grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
        aind_z = (np.minimum(np.maximum(kgrid_z,grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])
        w_mask = (aind_x == kgrid_x-grid_r[0,0])*(aind_y == kgrid_y-grid_r[1,0])*(aind_z == kgrid_z-grid_r[2,0])
        w = w*w_mask
        
        kn = 0
        for k in batch_ind:
            w_k = w[kn,:]
            data_ct[aind_x[kn],aind_y[kn],aind_z[kn]] += w_k*np.inner(w_k,data_c[aind_x[kn],aind_y[kn],aind_z[kn]])
            kn += 1
    return data_ct