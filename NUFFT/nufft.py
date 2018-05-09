import numpy as np
from FFT import fft
from NUFFT import kb128
from util.process_3d import zpad, crop, resize
from numba import jit
import NUFFT.nufft_util1 as nu1
import time

try:
    import cupy as cp
    CUDA_flag = True
except ImportError:
    CUDA_flag = False
    
# clean version
# non cartesian dims(limited to 5d):
#     data    : [1, N, 1, N_Parallel, N_Phase]
#     traj    : [3, N, 1, 1, N_Phase]
#     pattern : [1, N, 1, 1, N_Phase]
# 

kb_table = kb128.kb128
kb_table2 = kb128.kb128_2
Ndims = 6 # Data Dimension
Sdim = 3 # Spatial Dimension
Ddim = 3 # Data Dimension
Padim = 4 # Parallel Dimension
Phdim = 5 # Phase Dimension
dtpye_n = np.complex64



class NUFFT3D():
    def __init__(self, traj, grid_r = None, os = 1, pattern = None, width = 3, seg = 500000, Toeplitz_flag = False):
        # trajectory
        self.traj = os*traj
        if np.ndim(self.traj) < Ndims:
            for _ in range(Ndims-np.ndim(self.traj)):
                self.traj = np.expand_dims(self.traj,axis=-1)
                
        # Matrix Size        
        if grid_r is None:
            grid_L = np.abs(np.floor(np.min(self.traj.reshape([Sdim,-1]),axis=1)))
            grid_H = np.abs(np.ceil(np.max(self.traj.reshape([Sdim,-1]),axis=1)))
            grid_r = np.maximum(grid_L,grid_H)
            self.grid_r = (np.stack([-grid_r,grid_r],axis=1)).astype(np.int32)
            print('Est. Matrix size:',self.grid_r)
        else:
            self.grid_r = grid_r.astype(np.int32)
            
        # Data Parameter Def
        self.samples = np.prod((self.traj).shape[1:3])
        self.nPhase = np.prod(self.traj.shape[Phdim-1:])
        # bug
        self.nParallel = 1
        self.ndata_shape = [1,self.samples,1,self.nParallel,self.nPhase]
        self.traj_shape = [3,self.samples,1,1,self.nPhase]
        self.p_shape = [1,self.samples,1,1,self.nPhase]
        self.traj = self.traj.reshape(self.traj_shape)
        
        # Recon Parameter Def
        self.width = width
        self.seg = seg 
        self.I_size = [self.grid_r[i,1] - self.grid_r[i,0] for i in range(3)]
        self.cdata_shape = self.I_size + [self.nParallel,self.nPhase]
        self.p = None if pattern is None else np.abs(pattern.reshape(self.p_shape))

        # Function Def: Grid, GridH, Toeplitz
        self.A = fft.fft(shape=1,axes=(0,1,2))
        self.KB_win = self.A.IFT(KB_compensation(self.grid_r,self.I_size,width))
        self.KB_win  =self.KB_win[:,:,:,None,None]
        
        # Toeplitz mode prep
        if Toeplitz_flag:
            self.I_sizet = [x*2 for x in self.I_size]
            self.psf = self.Toeplitz_prep()
        else:
            self.psf = None
        
    def Toeplitz_prep(self):
        ndata_shapet = self.ndata_shape
        cdata_shapet = self.I_sizet + [self.nParallel,self.nPhase]
        ndata_shapet[Padim-1] = 1
        cdata_shapet[Padim-1] = 1
        
        os = 2
        # psf = np.zeros([self.I_size,1,self.nPhase])
        M = np.ones(ndata_shapet)
        if self.p is not None :
            M = M*self.p
        if CUDA_flag:
            psf_k = gridH_gpu(self.samples, ndata_shapet, cdata_shapet, self.traj*os, M, self.grid_r*os, self.width, self.seg)
        else:
            psf_k = gridH(self.samples, ndata_shapet, cdata_shapet, self.traj*os, M, self.grid_r*os, self.width, self.seg)
        
        return psf_k
        
        # Kernel Deconv
    def forward(self,img_c):
        tPa = img_c.shape[Padim-1] if img_c.ndim>=Padim else 1
        ndata_shapet = self.ndata_shape
        cdata_shapet = self.cdata_shape
        img_c = resize(img_c,tuple(self.I_size))
        print(img_c.shape,self.I_size)
        ndata_shapet[Padim-1] = tPa
        cdata_shapet[Padim-1] = tPa
        data_c = self.A.FT(img_c)
        if CUDA_flag:
            data_n = grid_gpu(self.samples, ndata_shapet, cdata_shapet, self.traj, data_c, self.grid_r, self.width, self.seg)
        else:
            data_n = grid(self.samples, ndata_shapet, cdata_shapet, self.traj, data_c, self.grid_r, self.width, self.seg)
 
        return data_n
        
    def adjoint(self,data_n):
        tPa = data_n.shape[Padim-1] if data_n.ndim>=Padim else 1
        ndata_shapet = self.ndata_shape
        cdata_shapet = self.cdata_shape
        ndata_shapet[Padim-1] = tPa
        cdata_shapet[Padim-1] = tPa
        data_n = np.reshape(data_n,self.ndata_shape)
        if self.p is not None :
            data_n = data_n*self.p
        if CUDA_flag:
            data_c = gridH_gpu(self.samples, ndata_shapet, cdata_shapet, self.traj, data_n, self.grid_r, self.width, self.seg)
        else:
            data_c = gridH(self.samples, ndata_shapet, cdata_shapet, self.traj, data_n, self.grid_r, self.width, self.seg)
        print(data_c.shape)
        img_hc = self.A.IFT(data_c)/self.KB_win
        
        return img_hc
    
    def Toeplitz(self,img_c):
        # faster testing 
        tPa = img_c.shape[Padim-1] if img_c.ndim>=Padim else 1
        cdata_shapet = self.cdata_shape
        cdata_shapet[Padim-1] = tPa
        img_c = np.reshape(img_c,cdata_shapet)
        img_c = resize(img_c,tuple(self.I_sizet))
        data_c = self.A.FT(img_c)
        data_c = data_c * self.psf
        img_ct = resize(self.A.IFT(data_c),tuple(self.I_size))
        return img_ct

    def density_est(self):
        # TODO add density estimation
        density = None
        return density

def KB_compensation(grid_r, I_size, width):
    kb_t = kb_table2
    win = np.zeros(I_size,dtype = np.complex64)
    c_ind = -grid_r[:,0]
    win_L = (np.ceil(-width)).astype(np.int32)
    win_H = (np.floor(width)+1).astype(np.int32)
    win_ind = (np.arange(win_L,win_H)).astype(np.int32)
    
    # kx,ky,kz same
    w = KB_weight(win_ind[None,:],kb_t,width).ravel()
    
    win_k = w[:,None,None]*w[None,:,None]*w[None,None,:]
    win[win_L+c_ind[0]:win_H+c_ind[0],win_L+c_ind[1]:win_H+c_ind[1],win_L+c_ind[2]:win_H+c_ind[2]] = win_k
    return win      
    
def KB_weight(grid, kb_2, width):
    # grid [N,2*width] kb_table[128]
    scale = (width)/(kb_2.shape[0]-1)
    frac = np.minimum(np.abs(grid)/scale,kb_table.size-1)
    (frac,grid_s) = np.modf(frac)
    
    w= np.sum(np.stack(((1-frac),frac),axis=2)*kb_2[grid_s.astype(int),:],axis=2)
    return w
       

def KB_weight_gpu(grid, kb_2, width):
    # grid [N,2*width] kb_table[128]
    scale = (width)/(kb_table.size-1)
    frac = cp.minimum(cp.abs(grid)/scale,kb_table.size-1)
    (frac,grid_s) = cp.modf(frac)
    
    w= cp.sum(cp.stack(((1-frac),frac),axis=2)*kb_2[grid_s.astype(int),:],axis=2)
    return w

@jit()
def grid_weight_calc(kx , ky, kz, grid_r, shape_stride, width):
    # kx,ky,kz: [N,1],non-scaled trajectory
    
    kb  = kb_table2
    batch_size = kx.shape[0]
    kernal_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size

    rind_x = np.round(kx+kernal_ind).astype(np.int32)
    rind_y = np.round(ky+kernal_ind).astype(np.int32)
    rind_z = np.round(kz+kernal_ind).astype(np.int32)

    wx = KB_weight(np.abs(rind_x-kx),kb,width)
    wy = KB_weight(np.abs(rind_y-ky),kb,width)
    wz = KB_weight(np.abs(rind_z-kz),kb,width)

    # N*x*y*z
    w = wx[:,:,None,None]*wy[:,None,:,None]*wz[:,None,None,:]

    # limit all the gridding points in the grid
    aind_x = (np.minimum(np.maximum(rind_x[:,:,None,None],grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])
    aind_y = (np.minimum(np.maximum(rind_y[:,None,:,None],grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])
    aind_z = (np.minimum(np.maximum(rind_z[:,None,None,:],grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])

    strides_ind = shape_stride[0]*aind_x + shape_stride[1]*aind_y + shape_stride[2]*aind_z

    w_mask = (aind_x == rind_x[:,:,None,None]-grid_r[0,0])*(aind_y == rind_y[:,None,:,None]-grid_r[1,0])*(aind_z == rind_z[:,None,None,:]-grid_r[2,0])

    w = w*w_mask
    w = np.reshape(w,[batch_size,-1]).astype(np.float32)
    strides_ind = np.reshape(strides_ind,[batch_size,-1])
    
    return w, strides_ind



def grid_gpu(samples, ndata_shape, cdata_shape, traj, data_c, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,nPh],non-scaled trajectory
    # data_c: [X,Y,Z,nPa,nPh],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: batch by batch gridding
    kb_g  = cp.asarray(kb_table2)
    grid_rc = cp.asarray(grid_r)
    
    # preparation
    # nCoil = cdata_shape[3]
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    shape_grid = cdata_shape[0:Sdim]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    kernal_ind = cp.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    
    # data cartesian
    data_n = np.zeros(ndata_shape,dtype = np.complex64)
    data_c = np.reshape(data_c,[np.prod(np.array(shape_grid)),1,nCoil,nPhase])
    #GPU domain
    t0 = time.time()
    # all dimensions change:
    # 5D [samples, 1, PChannels, 1, nPhase]
    for nP in range(nPhase):
        for i in range((samples-1)//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            batch_ind2 = cp.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            # load data into GPU from host

            kx = cp.asarray(traj[0,batch_ind,:,0,nP])
            ky = cp.asarray(traj[1,batch_ind,:,0,nP])
            kz = cp.asarray(traj[2,batch_ind,:,0,nP])

            rind_x = cp.rint(kx+kernal_ind).astype(cp.int32)
            rind_y = cp.rint(ky+kernal_ind).astype(cp.int32)
            rind_z = cp.rint(kz+kernal_ind).astype(cp.int32)

            wx = KB_weight_gpu(cp.abs(rind_x-kx),kb_g,width)
            wy = KB_weight_gpu(cp.abs(rind_y-ky),kb_g,width)
            wz = KB_weight_gpu(cp.abs(rind_z-kz),kb_g,width)

            # N*x*y*z
            w = wx[:,:,None,None]*wy[:,None,:,None]*wz[:,None,None,:]

            # limit all the gridding points in the grid
            aind_x = (cp.minimum(cp.maximum(rind_x[:,:,None,None],grid_rc[0,0]),grid_rc[0,1]-1) - grid_rc[0,0])
            aind_y = (cp.minimum(cp.maximum(rind_y[:,None,:,None],grid_rc[1,0]),grid_rc[1,1]-1) - grid_rc[1,0])
            aind_z = (cp.minimum(cp.maximum(rind_z[:,None,None,:],grid_rc[2,0]),grid_rc[2,1]-1) - grid_rc[2,0])

            w_mask = (aind_x == rind_x[:,:,None,None]-grid_rc[0,0])*(aind_y == rind_y[:,None,:,None]-grid_rc[1,0])*(aind_z == rind_z[:,None,None,:]-grid_rc[2,0])
            w = w*w_mask
            w = cp.reshape(w,[batch_ind.size,-1])
            
            strides_ind = shape_stride[0]*aind_x + shape_stride[1]*aind_y + shape_stride[2]*aind_z
            strides_ind = strides_ind.reshape([batch_ind.size,-1])
            strides_indt = cp.asnumpy(strides_ind)
            data_ct = cp.asarray(data_c[strides_indt,0,:,nP])
            data_n[0,batch_ind,0,:,nP] = cp.asnumpy(cp.sum(w.reshape([batch_ind.size,-1,1])*data_ct,axis=1))
            # data_ct = data_c[strides_indt,0,:,nP]
            # data_n[0,batch_ind,0,:,nP] = np.sum(cp.asnumpy(w.reshape([batch_ind.size,-1,1]))*data_ct,axis=1)


            # timing
            print('Grid time:',time.time()-t0)
    data_n = data_n.reshape([3,samples,1,nCoil,nPhase])
    return data_n


def gridH_gpu(samples, ndata_shape, cdata_shape, traj, data_n, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,nbin],non-scaled trajectory
    # data_n: [1,N,1,nC,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    # kb_t = kb_table
    kb_g  = cp.asarray(kb_table2)
    grid_rc = cp.asarray(grid_r)
    
    # preparation
    # nCoil = cdata_shape[3]
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    shape_grid = cdata_shape[0:Sdim]
    print(shape_grid)
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    kernal_ind = cp.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    
    # data cartesian
    data_c = np.zeros([np.prod(np.array(shape_grid)),1,nCoil,nPhase],dtype = np.complex64)
    data_n = np.reshape(data_n,ndata_shape)
    
    #GPU domain
    t0 = time.time()
    # all dimensions change:
    # 5D [samples, 1, PChannels, 1, nPhase]
    for nP in range(nPhase):
        
        for i in range((samples-1)//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))

            kx = cp.asarray(traj[0,batch_ind,:,0,nP])
            ky = cp.asarray(traj[1,batch_ind,:,0,nP])
            kz = cp.asarray(traj[2,batch_ind,:,0,nP])

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
            data_ci = cp.zeros([np.prod(np.array(shape_grid)),nCoil],dtype = np.float32)
            data_cr = cp.zeros([np.prod(np.array(shape_grid)),nCoil],dtype = np.float32)
            data_n_g = cp.asarray(data_n[0,batch_ind,:,:,nP])
            wdata_n = (w[:,:,None]*data_n_g).reshape([-1,nCoil])
            
            cp.scatter_add(data_ci,strides_ind,cp.imag(wdata_n))
            cp.scatter_add(data_cr,strides_ind,cp.real(wdata_n))
            data_c[:,0,:,nP] += cp.asnumpy(data_cr + 1j*data_ci)

            # timing
            print('Batch Grid time:',time.time()-t0)
    # back to host
    data_c = data_c.reshape(shape_grid+[nCoil,nPhase])
    
    return data_c

def gridH(samples, ndata_shape, cdata_shape, traj, data_n, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,1,nbin],non-scaled trajectory
    # data_n: [1,N,1,nC,1,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    # kb_t = kb_table
    # kb  = kb_table2
    
    # preparation
    # nCoil = cdata_shape[3]
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    shape_grid = cdata_shape[0:Sdim]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    kernal_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    
    # data cartesian
    data_c = np.zeros([np.prod(np.array(shape_grid)),nCoil,nPhase],dtype = np.complex64)
    data_n = np.reshape(data_n,ndata_shape)
    
    #GPU domain
    t0 = time.time()
    # all dimensions change:
    # 5D [samples, 1, PChannels, 1, nPhase]
    for nP in range(nPhase):   
        for i in range((samples-1)//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            kx = traj[0,batch_ind,:,0,nP]
            ky = traj[1,batch_ind,:,0,nP]
            kz = traj[2,batch_ind,:,0,nP]

            w, strides_ind = grid_weight_calc(kx, ky, kz, grid_r, shape_stride, width)

            # Coil Loop Memory limitation
            data_nt = data_n[0,batch_ind,:,:,nP]

            data_c[:,:,nP] = nu1.cadd2(data_c[:,:,nP],strides_ind,w,data_nt)
            print('Grid time:',time.time()-t0)
    # back to host
    data_c = data_c.reshape(shape_grid+[nCoil,nPhase])
    return data_c

def grid(samples, ndata_shape, cdata_shape, traj, data_c, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N],non-scaled trajectory
    # data_c: [Nx,Ny,Nz],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use

    # preparation
    kb_t = kb_table2
    # nCoil = cdata_shape[3]
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    shape_grid = cdata_shape[0:Sdim]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    kernal_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernal_ind = kernal_ind[None,:]
    k_len = kernal_ind.size
    
    # data cartesian
    data_n = np.zeros(ndata_shape,dtype = np.complex64)
    data_c = np.reshape(data_c,[np.prod(np.array(shape_grid)),1,nCoil,nPhase])
    #GPU domain
    t0 = time.time()
    # all dimensions change:
    # 5D [samples, 1, PChannels, 1, nPhase]
    for nP in range(nPhase):
        for i in range((samples-1)//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))

            kx = traj[0,batch_ind,:,0,nP]
            ky = traj[1,batch_ind,:,0,nP]
            kz = traj[2,batch_ind,:,0,nP]
            t1 = time.time()
            w, strides_ind = grid_weight_calc(kx, ky, kz, grid_r, shape_stride, width)
            print('weight time:',time.time()-t1)
            data_ct = data_c[:,:,:,nP]
            data_n[0,batch_ind,0,:,nP] = nu1.cadd3(data_n[0,batch_ind,0,:,nP],strides_ind,w,data_ct)
            # timing
            print('Grid time:',time.time()-t0)
    data_n = data_n.reshape([1,samples,1,nCoil,nPhase])
    return data_n
