import numpy as np
from FFT import fft
from NUFFT import kb128
from util.process_3d import zpad, crop, resize
from numba import jit
import NUFFT.nufft_util1 as nu1
import time

try:
    import cupy as cp
    from numba import cuda
    import NUFFT.nufft_util2 as nu2
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
dtype_c = np.complex64
dtype_r = np.float32


class NUFFT3D():
    def __init__(self, traj, grid_r = None, os = 1, pattern = None, width = 3, seg = 500000, Toeplitz_flag = False, CUDA_flag1 = False):
        self.cuda = CUDA_flag & CUDA_flag1
        # trajectory
        self.traj = os*traj.astype(dtype_r)
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
        self.traj = np.reshape(self.traj,self.traj_shape)

        # Recon Parameter Def
        self.width = width
        self.seg = seg 
        self.I_size = [self.grid_r[i,1] - self.grid_r[i,0] for i in range(3)]
        self.cdata_shape = self.I_size + [self.nParallel,self.nPhase]
        self.p = None if pattern is None else np.abs(pattern.astype(dtype_c).reshape(self.p_shape))

        # Function Def: Grid, GridH, Toeplitz
        if self.cuda:
            self.A = fft.fft_gpu(axes=(0,1,2))
        else:
            self.A = fft.fft(shape=1,axes=(0,1,2))
        self.KB_win = self.A.IFT(KB_compensation(self.grid_r,self.I_size,width + .5))
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
        if self.cuda:
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
        print(self.traj[2,:100,0,0,0])
        if self.cuda:
            data_n = grid_gpu(self.samples, ndata_shapet, cdata_shapet, self.traj, data_c, self.grid_r, self.width, self.seg)
        else:
            data_n = grid(self.samples, ndata_shapet, cdata_shapet, self.traj, data_c, self.grid_r, self.width, self.seg)
 
        return data_n
        
    def adjoint(self,data_n):
        if data_n.dtype is not dtype_c:
            print('tansfer to single precise type ...')
            data_n = data_n.astype(dtype_c)
        tPa = data_n.shape[Padim-1] if data_n.ndim>=Padim else 1
        ndata_shapet = self.ndata_shape
        cdata_shapet = self.cdata_shape
        ndata_shapet[Padim-1] = tPa
        cdata_shapet[Padim-1] = tPa
        data_n = np.reshape(data_n,self.ndata_shape)
        if self.p is not None :
            data_n = data_n*self.p

        if self.cuda:
            data_c = gridH_gpu(self.samples, ndata_shapet, cdata_shapet, self.traj, data_n, self.grid_r, self.width, self.seg)
        else:
            data_c = gridH(self.samples, ndata_shapet, cdata_shapet, self.traj, data_n, self.grid_r, self.width, self.seg)
        t0 = time.time()
        img_hc = self.A.IFT(data_c)/self.KB_win
        print('FFT time:',time.time()-t0)
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


@jit()
def KB_weight(grid, kb_2, width):
    # grid [N,2*width] kb_table[128]
    scale = (width)/(kb_2.shape[0]-1)
    frac = np.minimum(np.abs(grid)/scale,kb_table.size-1)
    (frac,grid_s) = np.modf(frac)
    
    w= np.sum(np.stack(((1-frac),frac),axis=2)*kb_2[grid_s.astype(int),:],axis=2)
    return w

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
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    # data cartesian
    data_c = np.zeros(cdata_shape,dtype = np.complex64)
    data_n = np.reshape(data_n,ndata_shape)

    scale = (width+.5)/(kb_table.shape[0]-1)
    
    t0 = time.time()
    for nP in range(nPhase):
        for i in range((samples-1)//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            traj1 = traj[:,batch_ind,0,0,nP]
            data_nt = data_n[0,batch_ind,0,:,nP]
            nu1.gridH2(data_c[:,:,:,:,nP], data_nt, traj1, np.array(grid_r), width, scale, kb_table)
    print('Total Griding time:',time.time()-t0)
    # data_c = data_c.reshape(cdata_shape)
    return data_c

def gridH_gpu(samples, ndata_shape, cdata_shape, traj, data_n, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,1,nbin],non-scaled trajectory
    # data_n: [1,N,1,nC,1,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    # 
    # kb  = kb_table2
    
    # preparation
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    # data cartesian
    data_c = np.zeros(cdata_shape,dtype = dtype_c)
    data_n = np.reshape(data_n,ndata_shape)
    
    data_cr = np.zeros(cdata_shape[:-1],dtype = dtype_r)
    data_ci = np.zeros(cdata_shape[:-1],dtype = dtype_r)
    
    scale = (width+.5)/(kb_table.shape[0]-1)
    
    t0 = time.time()
    for nP in range(nPhase):
        data_cr[:] = 0
        data_ci[:] = 0
        for i in range((samples-1)//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            traj1 = np.ascontiguousarray(traj[:,batch_ind,0,0,nP])
            data_nr = np.ascontiguousarray(np.real(data_n[0,batch_ind,0,:,nP]))
            data_ni = np.ascontiguousarray(np.imag(data_n[0,batch_ind,0,:,nP]))

            tpb = (16,8)
            bpg = (int(np.ceil(batch_size/tpb[0])),int(np.ceil(nCoil/tpb[1])))
            nu2.gridH2_gpu[bpg,tpb](data_cr, data_ci, data_nr, data_ni, traj1, np.array(grid_r), width, scale, kb_table)
        
        data_c[:,:,:,:,nP] = data_cr +1j*data_ci 
    print('Total Griding time:',time.time()-t0)
    return data_c

def grid(samples, ndata_shape, cdata_shape, traj, data_c, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,1,nbin],non-scaled trajectory
    # data_n: [1,N,1,nC,1,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    # kb_t = kb_table
    # kb  = kb_table2
    
    # preparation
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    # data cartesian
    data_c = np.zeros(cdata_shape,dtype = np.complex64)
    data_n = np.reshape(data_n,ndata_shape)

    scale = (width+.5)/(kb_table.shape[0]-1)
    for nP in range(nPhase):
        for i in range((samples-1)//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            traj1 = traj[:,batch_ind,0,0,nP]
            nu1.grid2(data_n[0,batch_ind,0,:,nP], data_c[:,:,:,:,nP], traj1, np.array(grid_r), width, scale, kb_table)
    print('Total Griding time:',time.time()-t0)
    # data_c = data_c.reshape(cdata_shape)
    return data_n

def grid_gpu(samples, ndata_shape, cdata_shape, traj, data_c, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,1,nbin],non-scaled trajectory
    # data_n: [1,N,1,nC,1,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    # kb_t = kb_table
    # kb  = kb_table2
    
    # preparation
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    # data cartesian
    data_c = np.zeros(cdata_shape,dtype = np.complex64)
    data_n = np.reshape(data_n,ndata_shape)
    
    scale = (width+.5)/(kb_table.shape[0]-1)
    
    t0 = time.time()
    for nP in range(nPhase):
        data_cr = np.ascontiguousarray(np.real(data_c[:,:,:,:,nP]))
        data_ci = np.ascontiguousarray(np.imag(data_c[:,:,:,:,nP]))
        for i in range((samples-1)//batch_size + 1):
            batch_ind = np.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            traj1 = np.ascontiguousarray(traj[:,batch_ind,0,0,nP])
            # check if it is contiguous
            data_nr = np.zeros((batch_ind.size,nCoil),dtype = dtype_r)
            data_ni = np.zeros((batch_ind.size,nCoil),dtype = dtype_r)
            
            tpb = (16,8)
            bpg = (int(np.ceil(batch_size/tpb[0])),int(np.ceil(nCoil/tpb[1])))
            nu2.gridH2_gpu[bpg,tpb](data_nr, data_ni, data_cr, data_ci, traj1, np.array(grid_r), width, scale, kb_table)
            data_n[0,batch_ind,0,:,nP] = data_nr +1j*data_ni 
    print('Total Griding time:',time.time()-t0)
    return data_n
    

def gridt(samples, ndata_shape, cdata_shape, traj, data_c, grid_r, width, batch_size, pattern = None):
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
    kernel_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernel_ind = kernel_ind[None,:]
    k_len = kernel_ind.size
    
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
