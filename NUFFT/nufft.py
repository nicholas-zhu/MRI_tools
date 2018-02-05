import numpy as np
import NUFFT
from FFT import fft
from NUFFT import kb128
from numba import jit
import time

# grid, gridH operators
# non cartesian dims:
#     1*N(samples)
#     3*N(trajectory)
#     TODO parallel
# input:
#     traj   : non-cartesian traj
#     grid   : cartesian grid
#     data_c : cartesian data
# output:
#     data_n : non_cartesian data

kb_table = kb128.kb128

class NUFFT3D():
    def __init__(self, traj, grid_r = None, os = 1.2, pattern = None, width = 2):
        self.traj = os*np.reshape(traj,[3,-1])
        self.samples = (self.traj).shape[1]
        if grid_r is None:
            self.grid_r = (np.stack([np.floor(np.min(self.traj,axis=1))-width, np.ceil(np.max(self.traj,axis=1))+width],axis=1)).astype(np.int32)
            grid_L = np.abs(np.floor(np.min(self.traj,axis=1))-width)
            grid_H = np.abs(np.ceil(np.max(self.traj,axis=1))+width)
            grid_r = np.maximum(grid_L,grid_H)
            self.grid_r = (np.stack([-grid_r,grid_r],axis=1)).astype(np.int32)
            print('Est. kspace size:',self.grid_r)
        else:
            self.grid_r = grid_r.astype(np.int32)
        if pattern is None:
            self.p = None
        else:
            self.p = np.reshape(pattern,[1,-1])
        
        self.A = fft.fft(shape=1,axes=(0,1,2))
        self.width = width
        self.KB_win = self.A.IFT(KB_compensation(self.grid_r,width))
        # kb psf compensation
        
    def forward(self,img_c):
        data_c = self.A.FT(img_c)
        data_n = grid(self.samples,self.traj, data_c, self.grid_r, width =self.width)
        
        return data_n
        
    def adjoint(self,data_n,w_flag = False):
        data_n = np.reshape(data_n,[1,-1])
        if self.p is None :
            data_c = gridH(self.samples,self.traj, data_n, self.grid_r, width =self.width)
        else:
            data_c = gridH(self.samples,self.traj, data_n*self.p, self.grid_r, width = self.width)
        img_hc = self.A.IFT(data_c)
        return img_hc
    
    def Toeplitz(self,img_c):
        data_c = self.A.FT(img_c)
        data_ct = gTg2(self.samples,self.traj, data_c, self.grid_r, self.width, pattern = self.p)
        img_ct = self.A.IFT(data_ct)
        return img_ct

    def density_est(self):
        # TODO add density estimation
        density = None
        return density

    
def KB_compensation(grid_r, width):
    kb_t = kb_table
    win = np.zeros([grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]],dtype = np.complex128)
    c_ind = -grid_r[:,0]
    win_L = (np.ceil(-width)).astype(np.int32)
    win_H = (np.floor(width)+1).astype(np.int32)
    win_ind = (np.arange(win_L,win_H)).astype(np.int32)
    
    # kx,ky,kz same
    w = KB_weight(win_ind[None,:],kb_t,width).ravel()
    
    win_k = w[:,None,None]*w[None,:,None]*w[None,None,:]
    win[win_L+c_ind[0]:win_H+c_ind[0],win_L+c_ind[1]:win_H+c_ind[1],win_L+c_ind[2]:win_H+c_ind[2]] = win_k
    return win

@jit(nopython=True)
def gridT_sum(data_t,index_t,data_s,weight,pattern,N):
    # numba based Toepliz gridding
    # data_t: [N_t*1] target data
    # index_s: [N_s*L] sample to target data index
    # data_s: [N_s*L] sample data
    # weight: [N_s*L] KB_win
    # pattern: [N_index] density

    if pattern.size == 1:
        for i in range(N):
            index = index_t[i,:]
            data_t[index] += np.sum(data_s[i,:]*weight[i,:])*weight[i,:]
    else:
        for i in range(N):
            index = index_t[i,:]
            data_t[index] += np.sum(data_s[i,:]*weight[i,:])*weight[i,:]*pattern[i]
        
    return data_t     
    
@jit(nopython=True)
def gridH_sum(data_t,index_s,data_s,N):
    # numba based gridding
    # data_t: [N_t*1] target data
    # index_s: [N_s*1] sample to target data index
    # data_s: [N_s*1) sample data
    for i in range(N):
        data_t[index_s[i]] += data_s[i]
        
    return data_t   

def KB_weight(grid, kb_table, width):
    # grid [N,2*width] kb_table[128]
    scale = (width)/(kb_table.size-1)
    frac = np.minimum(np.abs(grid)/scale,kb_table.size-2)
    (frac,grid_s) = np.modf(frac)
    
    # shift table [KB(n),KB(n+1)]
    kb_2 = np.stack((kb_table,np.roll(kb_table,-1,axis=0)),axis=1)
    kb_2[-1,1]=0
    
    w= np.sum(np.stack(((1-frac),frac),axis=2)*kb_2[grid_s.astype(int),:],axis=2)
    return w
    
def gridH(samples, traj, data_n, grid_r, width, batch_size = 1000000):
    # samples: int(N), num of sample
    # traj: [3,N],non-scaled trajectory
    # data_n: [1,N],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    data_n = data_n.ravel()
    kb_t = kb_table
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:]
    
    shape_grid = [grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    data_c = np.zeros(np.prod(shape_grid),dtype = np.complex128)
    
    kernal_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    
    for i in range(samples//batch_size + 1):
        t0 = time.time()
        batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
        rind_x = np.round(kx[batch_ind][:,None]+kernal_ind).astype(np.int32)
        rind_y = np.round(ky[batch_ind][:,None]+kernal_ind).astype(np.int32)
        rind_z = np.round(kz[batch_ind][:,None]+kernal_ind).astype(np.int32)
        
        wx = KB_weight(np.abs(rind_x-kx[batch_ind][:,None]),kb_t,width)
        wy = KB_weight(np.abs(rind_y-ky[batch_ind][:,None]),kb_t,width)
        wz = KB_weight(np.abs(rind_z-kz[batch_ind][:,None]),kb_t,width)
        
        # N*x*y*z
        w = wx[:,:,None,None]*wy[:,None,:,None]*wz[:,None,None,:]
        
        # limit all the gridding points in the grid
        aind_x = (np.minimum(np.maximum(rind_x[:,:,None,None],grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
        aind_y = (np.minimum(np.maximum(rind_y[:,None,:,None],grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
        aind_z = (np.minimum(np.maximum(rind_z[:,None,None,:],grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])
        #w_mask = (aind_x == rind_x[:,:,None,None]-grid_r[0,0])*(aind_y == rind_y[:,None,:,None]-grid_r[1,0])*(aind_z == rind_z[:,None,None,:]-grid_r[2,0])
        #w = w*w_mask

        strides_ind = shape_stride[0]*aind_x + shape_stride[1]*aind_y + shape_stride[2]*aind_z
        strides_ind = strides_ind.ravel()
        wdata_n = (w*data_n[batch_ind][:,None,None,None]).ravel()
        
        #np.add.at(data_c,v_ind,wdata_n)
        gridH_sum(data_c,strides_ind,wdata_n,strides_ind.size)
        print('Batch Grid time:',time.time()-t0)
        
    data_c = data_c.reshape(shape_grid)
    return data_c

def grid(samples, traj, data_c, grid_r, width, batch_size = 500000):
    # samples: int(N), num of sample
    # traj: [3,N],non-scaled trajectory
    # data_c: [Nx,Ny,Nz],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    data_c = data_c.ravel()
    kb_t = kb_table
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:]
    
    shape_grid = [grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    data_n = np.zeros([1,samples],dtype = np.complex128)
    
    kernal_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    for i in range(samples//batch_size + 1):
        t0 = time.time()
        batch_size = batch_ind.size
        batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
        rind_x = np.round(kx[batch_ind][:,None]+kernal_ind).astype(np.int32)
        rind_y = np.round(ky[batch_ind][:,None]+kernal_ind).astype(np.int32)
        rind_z = np.round(kz[batch_ind][:,None]+kernal_ind).astype(np.int32)
        
        wx = KB_weight(np.abs(rind_x-kx[batch_ind][:,None]),kb_t,width)
        wy = KB_weight(np.abs(rind_y-ky[batch_ind][:,None]),kb_t,width)
        wz = KB_weight(np.abs(rind_z-kz[batch_ind][:,None]),kb_t,width)
        
        # N*x*y*z
        w = wx[:,:,None,None]*wy[:,None,:,None]*wz[:,None,None,:]
        
        # limit all the gridding points in the grid
        aind_x = (np.minimum(np.maximum(rind_x[:,:,None,None],grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
        aind_y = (np.minimum(np.maximum(rind_y[:,None,:,None],grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
        aind_z = (np.minimum(np.maximum(rind_z[:,None,None,:],grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])
        w_mask = (aind_x == rind_x[:,:,None,None]-grid_r[0,0])*(aind_y == rind_y[:,None,:,None]-grid_r[1,0])*(aind_z == rind_z[:,None,None,:]-grid_r[2,0])
        w = w*w_mask
        
        strides_ind = shape_stride[0]*aind_x + shape_stride[1]*aind_y + shape_stride[2]*aind_z
        strides_ind = strides_ind.reshape([batch_size,-1])
        data_n[0,batch_ind] = np.sum(w.reshape([batch_size,-1])*data_c[strides_ind],axis=1)
        
        print('Batch Grid time:',time.time()-t0)
        
    return data_n

def gTg2(samples, traj, data_c, grid_r, width, pattern = None, batch_size = 500000):
    # samples: int(N), num of sample
    # traj: [3,N],non-scaled trajectory
    # data_c: [Nx,Ny,Nz],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    data_c = data_c.ravel()
    kb_t = kb_table
    kx = traj[0,:]
    ky = traj[1,:]
    kz = traj[2,:]
    
    shape_grid = [grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    data_ct = np.zeros_like(data_c)
    
    kernal_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    for i in range(samples//batch_size + 1):
        t0 = time.time()
        batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
        batch_size = batch_ind.size
        rind_x = np.round(kx[batch_ind][:,None]+kernal_ind).astype(np.int32)
        rind_y = np.round(ky[batch_ind][:,None]+kernal_ind).astype(np.int32)
        rind_z = np.round(kz[batch_ind][:,None]+kernal_ind).astype(np.int32)
        
        wx = KB_weight(np.abs(rind_x-kx[batch_ind][:,None]),kb_t,width)
        wy = KB_weight(np.abs(rind_y-ky[batch_ind][:,None]),kb_t,width)
        wz = KB_weight(np.abs(rind_z-kz[batch_ind][:,None]),kb_t,width)
        
        # N*x*y*z
        w = wx[:,:,None,None]*wy[:,None,:,None]*wz[:,None,None,:]
        
        # limit all the gridding points in the grid
        aind_x = (np.minimum(np.maximum(rind_x[:,:,None,None],grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
        aind_y = (np.minimum(np.maximum(rind_y[:,None,:,None],grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
        aind_z = (np.minimum(np.maximum(rind_z[:,None,None,:],grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])
        w_mask = (aind_x == rind_x[:,:,None,None]-grid_r[0,0])*(aind_y == rind_y[:,None,:,None]-grid_r[1,0])*(aind_z == rind_z[:,None,None,:]-grid_r[2,0])
        w = w*w_mask
        w = w.reshape([batch_size,-1])
        
        strides_ind = shape_stride[0]*aind_x + shape_stride[1]*aind_y + shape_stride[2]*aind_z
        strides_ind = strides_ind.reshape([batch_size,-1])
        
        data_s = data_c[strides_ind]
        if pattern is None:
            gridT_sum(data_ct,strides_ind,data_s,w,np.array([1]),batch_size)
        else:
            gridT_sum(data_ct,strides_ind,data_s,w,pattern,batch_size)
    
    data_ct = data_ct.reshape(shape_grid)
    return data_ct