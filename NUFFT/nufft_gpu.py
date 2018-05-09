import numpy as np
import NUFFT
from FFT import fft
from NUFFT import kb128
from numba import jit
import time

try:
    import cupy as cp
    CUDA_flag = True
except ImportError:
    import sqlite3
    CUDA_flag = False

# NUFFT GPU VERSION
# grid, gridH, Toeplitz
# non cartesian dims:
#     [1, N, 1, channel, 1, Phase](samples)
#     [3, N, 1, 1, 1, Phase](trajectory)
#     TODO parallel
# input:
#     traj   : non-cartesian traj
#     grid   : cartesian grid
#     pattern: weight
#


kb_table = kb128.kb128
kb_table2 = kb128.kb128_2
Ndim = 6 # Image Dimension
Pdim = 4 # Parallel Dimension
Sdim = 3 # Spatial Dimension

class NUFFT3D():
    def __init__(self, traj, grid_r = None, os = 1, pattern = None, width = 2, seg = 500000):
        # trajectory
        self.traj = os*traj
        if np.ndim(self.traj) < Ndim:
            for _ in range(Ndim-np.ndim(self.traj)):
                self.traj = np.expand_dims(self.traj,axis=-1)
        
        # gridding kspace
        if grid_r is None:
            grid_L = np.abs(np.floor(np.min(self.traj.reshape([Sdim,-1]),axis=1)))
            grid_H = np.abs(np.ceil(np.max(self.traj.reshape([Sdim,-1]),axis=1)))
            grid_r = np.maximum(grid_L,grid_H)
            self.grid_r = (np.stack([-grid_r,grid_r],axis=1)).astype(np.int32)
            print('Est. kspace size:',self.grid_r)
        else:
            self.grid_r = grid_r.astype(np.int32)
        
        # dimentions:
        self.nPhase = np.prod(self.traj.shape[Ndim-1:])
        # ncart
        # self.nchannel = 1 decided by data
        self.samples = np.prod((self.traj).shape[1:3])
        self.width = width
        self.seg = seg # 
        self.traj = np.reshape(self.traj,(Sdim, self.samples, 1, 1, 1, self.nPhase))
        # cart
        self.I_size = self.grid_r[:,1] - self.grid_r[:,0]
        
        # operator def
        self.A = fft.fft(shape=1,axes=(0,1,2))
        self.KB_win = self.A.IFT(KB_compensation(self.grid_r,width))
        self.KB_win  =self.KB_win[:,:,:,None,None,None]
        # kb psf compensation
        if pattern is None:
            self.p = None
        else:
            self.p = np.abs(pattern)
            self.p = np.reshape(self.p,(1, self.samples, 1, 1, 1, self.nPhase))
            
    def forward(self,img_c):
        
        data_c = self.A.FT(img_c)
        data_n = grid_gpu(self.samples,self.traj, data_c, self.grid_r, self.width, self.seg)
        
        return data_n
        
    def adjoint(self,data_n,w_flag = False):
        data_n = np.reshape(data_n,[1,self.samples,1,-1,1,self.nPhase])
        if self.p is None :
            data_c = gridH_gpu(self.samples,self.traj, data_n, self.grid_r, self.width, self.seg)
        else:
            print(data_n.shape,self.p.shape);
            data_nw = data_n*self.p
            data_c = gridH_gpu(self.samples,self.traj, data_nw, self.grid_r, self.width, self.seg)
            
        img_hc = self.A.IFT(data_c)/self.KB_win
        
        return img_hc
    
    def Toeplitz(self,img_c):
        # faster testing 
        data_c = self.A.FT(img_c)
        data_ct = gTg_gpu(self.samples,self.traj, data_c, self.grid_r, self.width, self.seg, self.p)
        img_ct = self.A.IFT(data_ct)
        return img_ct

    def density_est(self):
        # TODO add density estimation
        density = None
        return density

def KB_weight(grid, kb_2, width):
    # grid [N,2*width] kb_table[128]
    scale = (width)/(kb_2.shape[0]-1)
    frac = np.minimum(np.abs(grid)/scale,kb_table.size-2)
    (frac,grid_s) = np.modf(frac)
    
    
    w= np.sum(np.stack(((1-frac),frac),axis=2)*kb_2[grid_s.astype(int),:],axis=2)
    return w
    
def KB_compensation(grid_r, width):
    kb_t = kb_table2
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

def KB_weight_gpu(grid, kb_2, width):
    # grid [N,2*width] kb_table[128]
    scale = (width)/(kb_table.size-1)
    frac = cp.minimum(cp.abs(grid)/scale,kb_table.size-2)
    (frac,grid_s) = cp.modf(frac)
    
    w= cp.sum(cp.stack(((1-frac),frac),axis=2)*kb_2[grid_s.astype(int),:],axis=2)
    return w

def gridH_gpu(samples, traj, data_n, grid_r, width, batch_size):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,1,nbin],non-scaled trajectory
    # data_n: [1,N,1,nC,1,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    # kb_t = kb_table
    kb_g  = cp.asarray(kb_table2)
    
    # preparation
    nCoil = data_n.shape[3]
    nPhase = np.prod(data_n.shape[Pdim:]).astype(np.int32)
    assert nPhase == traj.shape[Pdim], " Data and Trajectory nPhase mismatch "
    assert samples == data_n.shape[1] * data_n.shape[2], " Data and Trajectory sample mismatch "
    shape_grid = [grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    kernal_ind = cp.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    
    # data cartesian
    data_c = np.zeros([np.prod(np.array(shape_grid)),nCoil,nPhase],dtype = np.complex64)
    
    #GPU domain
    t0 = time.time()
    # all dimensions change:
    # 5D [samples, 1, PChannels, 1, nPhase]
    for nP in range(nPhase):
        
        for i in range(samples//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))

            kx = cp.asarray(traj[0,batch_ind,:,0,0,nP])
            ky = cp.asarray(traj[1,batch_ind,:,0,0,nP])
            kz = cp.asarray(traj[2,batch_ind,:,0,0,nP])

            rind_x = cp.rint(kx+kernal_ind).astype(cp.int32)
            rind_y = cp.rint(ky+kernal_ind).astype(cp.int32)
            rind_z = cp.rint(kz+kernal_ind).astype(cp.int32)

            wx = KB_weight_gpu(cp.abs(rind_x-kx),kb_g,width)
            wy = KB_weight_gpu(cp.abs(rind_y-ky),kb_g,width)
            wz = KB_weight_gpu(cp.abs(rind_z-kz),kb_g,width)

            # N*x*y*z
            w = wx[:,:,None,None]*wy[:,None,:,None]*wz[:,None,None,:]

            # limit all the gridding points in the grid
            aind_x = (cp.minimum(cp.maximum(rind_x[:,:,None,None],grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
            aind_y = (cp.minimum(cp.maximum(rind_y[:,None,:,None],grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
            aind_z = (cp.minimum(cp.maximum(rind_z[:,None,None,:],grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])

            strides_ind = shape_stride[0]*aind_x + shape_stride[1]*aind_y + shape_stride[2]*aind_z
            strides_ind = strides_ind.ravel()
            w_mask = (aind_x == rind_x[:,:,None,None]-grid_r[0,0])*(aind_y == rind_y[:,None,:,None]-grid_r[1,0])*(aind_z == rind_z[:,None,None,:]-grid_r[2,0])
            
            w = w*w_mask
            w = cp.reshape(w,[batch_ind.size,-1]).astype(np.float32)
            # Coil Loop Memory limitation
            for nC in range(nCoil):
                data_ci = cp.zeros([np.prod(np.array(shape_grid))],dtype = np.float32)
                data_cr = cp.zeros([np.prod(np.array(shape_grid))],dtype = np.float32)
                # load data into GPU from host
                data_n_g = cp.asarray(data_n[0,batch_ind,:,nC,0,nP])
                wdata_n = (w*data_n_g).ravel()

                cp.scatter_add(data_ci,strides_ind,cp.imag(wdata_n))
                cp.scatter_add(data_cr,strides_ind,cp.real(wdata_n))

                # back to host
                data_c[:,nC,nP] += cp.asnumpy(data_cr + 1j*data_ci)
            # timing
            print('Batch Grid time:',time.time()-t0)
    # back to host
    data_c = data_c.reshape(shape_grid+[nCoil,1,nPhase])
    return data_c

def grid_gpu(samples, traj, data_c, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,1,nbin],non-scaled trajectory
    # data_c: [X,Y,Z,nC,1,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    kb_g  = cp.asarray(kb_table2)
    
    # preparation
    nCoil = data_c.shape[3]
    nPhase = np.prod(data_c.shape[Pdim:]).astype(np.int32)
    assert nPhase == traj.shape[Pdim], " Data and Trajectory nPhase mismatch "
    shape_grid = [grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    kernal_ind = cp.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    
    # data cartesian
    data_n = np.zeros([1,samples,1,nCoil,nPhase],dtype = np.complex64)
    data_c = np.reshape(data_c,[np.prod(np.array(shape_grid)),1,nCoil,nPhase])
    #GPU domain
    t0 = time.time()
    # all dimensions change:
    # 5D [samples, 1, PChannels, 1, nPhase]
    for nP in range(nPhase):
        for i in range(samples//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            batch_ind2 = cp.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            # load data into GPU from host
            

            kx = cp.asarray(traj[0,batch_ind,:,0,0,nP])
            ky = cp.asarray(traj[1,batch_ind,:,0,0,nP])
            kz = cp.asarray(traj[2,batch_ind,:,0,0,nP])

            rind_x = cp.rint(kx+kernal_ind).astype(cp.int32)
            rind_y = cp.rint(ky+kernal_ind).astype(cp.int32)
            rind_z = cp.rint(kz+kernal_ind).astype(cp.int32)

            wx = KB_weight_gpu(cp.abs(rind_x-kx),kb_g,width)
            wy = KB_weight_gpu(cp.abs(rind_y-ky),kb_g,width)
            wz = KB_weight_gpu(cp.abs(rind_z-kz),kb_g,width)

            # N*x*y*z
            w = wx[:,:,None,None]*wy[:,None,:,None]*wz[:,None,None,:]

            # limit all the gridding points in the grid
            aind_x = (cp.minimum(cp.maximum(rind_x[:,:,None,None],grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
            aind_y = (cp.minimum(cp.maximum(rind_y[:,None,:,None],grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
            aind_z = (cp.minimum(cp.maximum(rind_z[:,None,None,:],grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])

            w_mask = (aind_x == rind_x[:,:,None,None]-grid_r[0,0])*(aind_y == rind_y[:,None,:,None]-grid_r[1,0])*(aind_z == rind_z[:,None,None,:]-grid_r[2,0])
            w = w*w_mask
            w = cp.reshape(w,[batch_ind.size,-1])
            
            strides_ind = shape_stride[0]*aind_x + shape_stride[1]*aind_y + shape_stride[2]*aind_z
            strides_ind = strides_ind.reshape([batch_ind.size,-1])
            strides_indt = cp.asnumpy(strides_ind)
            data_ct = data_c[strides_indt,0,:,nP]
            data_n[0,batch_ind,0,:,nP] = np.sum(cp.asnumpy(w.reshape([batch_ind.size,-1,1]))*data_ct,axis=1)


            # timing
            print('Grid time:',time.time()-t0)
    data_n = data_n.reshape([3,samples,1,nCoil,1,nPhase])
    return data_n

def gTg2_gpu( index_t,data_s,weight,pattern,N):
    # gpu based Toepliz gridding
    # data_t: [N_t*1] target data
    # index_t: [N_s*L] sample to target data index
    # data_s: [N_s,1,nParallel] sample data
    # weight: [N_s*L] KB_win
    # pattern: [N_index] density
    #     data_t = WT*W*data_s
    #     6G GPU limitation
    
    data_t = np.zeros_like(data_s)
    index_c = cp.asnumpy(index_t)
    if pattern is not None:
        pattern_t = cp.asarray(pattern).reshape([-1,1])
    
    for i in range(data_s.shape[2]):
        data_ci = cp.zeros(data_s.shape[0],dtype=np.float32)
        data_cr = cp.zeros(data_s.shape[0],dtype=np.float32)
        data_st = cp.asarray(data_s[index_c,0,i],dtype = np.complex64)# [N_nc,L]
        data_st = cp.sum(data_st*weight,axis = 1,keepdims = True)
        if pattern is not None:
            data_st = data_st*pattern_t
        data_st = data_st*weight
        
        cp.scatter_add(data_ci,index_t,cp.imag(data_st))
        cp.scatter_add(data_cr,index_t,cp.real(data_st))
        
        data_t[:,0,i] = cp.asnumpy(data_ci+1j*data_cr)
        
    return data_t


def gTg3_gpu( index_t,data_s,weight,pattern,N):
    # gpu based Toepliz gridding
    # data_t: [N_t*1] target data
    # index_s: [N_s*L] sample to target data index
    # data_s: [N_s,1,nParallel] sample data
    # weight: [N_s*L] KB_win
    # pattern: [N_index] density
    
    data_t = np.zeros_like(data_s)
    index_c = cp.asnumpy(index_t)
    if pattern.size == 1:
        for i in range(N):
            index = index_c[i,:]
            data_st = cp.asarray(data_s[i,:,:])
            wt = weight[i,:][:,None]
            data_t[index,:] += cp.asnumpy(cp.sum(data_st*wt,axis=0)*wt)
    else:
        for i in range(N):
            index = index_c[i,:]
            data_st = cp.asarray(data_s[i,:,:])
            wt = weight[i,:][:,None]
            data_t[index,:,:] += cp.asnumpy(cp.sum(data_st*wt,axis=0)*wt)*pattern[i]
        
    return data_t

def gTg_gpu(samples, traj, data_c, grid_r, width, batch_size, pattern):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,1,nbin],non-scaled trajectory
    # data_c: [X,Y,Z,nC,1,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    # pattern:[1,N,1,1,1,nbin]
    
    
    kb_g  = cp.asarray(kb_table2)
    
    # preparation
    nCoil = data_c.shape[3]
    nPhase = np.prod(data_c.shape[Pdim:]).astype(np.int32)
    assert nPhase == traj.shape[Pdim], " Data and Trajectory nPhase mismatch "
    if pattern is not None:
        assert traj.shape[1:] == pattern.shape[1:], " Trajectory and Pattern shape mismatch"
        pattern = pattern.reshape([samples,1,nPhase])
        
    shape_grid = [grid_r[0,1]-grid_r[0,0],grid_r[1,1]-grid_r[1,0],grid_r[2,1]-grid_r[2,0]]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    kernal_ind = cp.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    
    # data cartesion
    data_c = np.reshape(data_c,[np.prod(np.array(shape_grid)),1,nCoil,nPhase])
    data_c = np.reshape(data_c,[np.prod(np.array(shape_grid)),1,nCoil,nPhase])
    data_ct = np.zeros_like(data_c)
    
    t0 = time.time()
    for nP in range(nPhase):
        for i in range(samples//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            # batch_ind2 = cp.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            # load data into GPU from host
            

            kx = cp.asarray(traj[0,batch_ind,:,0,0,nP])
            ky = cp.asarray(traj[1,batch_ind,:,0,0,nP])
            kz = cp.asarray(traj[2,batch_ind,:,0,0,nP])

            rind_x = cp.rint(kx+kernal_ind).astype(cp.int32)
            rind_y = cp.rint(ky+kernal_ind).astype(cp.int32)
            rind_z = cp.rint(kz+kernal_ind).astype(cp.int32)

            wx = KB_weight_gpu(cp.abs(rind_x-kx),kb_g,width)
            wy = KB_weight_gpu(cp.abs(rind_y-ky),kb_g,width)
            wz = KB_weight_gpu(cp.abs(rind_z-kz),kb_g,width)

            # N*x*y*z
            w = wx[:,:,None,None]*wy[:,None,:,None]*wz[:,None,None,:]
            # w = cp.reshape(w,(batch_ind.size,-1))

            # limit all the gridding points in the grid
            aind_x = (cp.minimum(cp.maximum(rind_x[:,:,None,None],grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
            aind_y = (cp.minimum(cp.maximum(rind_y[:,None,:,None],grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
            aind_z = (cp.minimum(cp.maximum(rind_z[:,None,None,:],grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])

            w_mask = (aind_x == rind_x[:,:,None,None]-grid_r[0,0])*(aind_y == rind_y[:,None,:,None]-grid_r[1,0])*(aind_z == rind_z[:,None,None,:]-grid_r[2,0])
            w = w*w_mask
            
            w = cp.reshape(w,[batch_ind.size,-1])
            
            strides_ind = shape_stride[0]*aind_x + shape_stride[1]*aind_y + shape_stride[2]*aind_z
            print(strides_ind.shape,w.shape,batch_ind.size,aind_z.shape)
            strides_ind = strides_ind.reshape([batch_ind.size,-1])
            
            if pattern is None:
                data_ct[:,:,:,nP] += gTg2_gpu(strides_ind,data_c[:,:,:,nP],w,np.array([1]),batch_ind.size)
            else:
                data_ct[:,:,:,nP] += gTg2_gpu(strides_ind,data_c[:,:,:,nP],w,pattern[batch_ind,nP],batch_ind.size)
            print('Batch Toepliz time:',time.time()-t0)
    data_ct = data_ct.reshape(shape_grid+[nCoil,1,nPhase])
    return data_ct
                
