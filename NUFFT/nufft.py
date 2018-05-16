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
            data_c = gridH_gpu1(self.samples, ndata_shapet, cdata_shapet, self.traj, data_n, self.grid_r, self.width, self.seg)
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
       

def KB_weight_gpu(grid, kb_2, width):
    # grid [N,2*width] kb_table[128]
    scale = (width)/(kb_table.size-1)
    frac = cp.minimum(cp.abs(grid)/scale,kb_table.size-1)
    (frac,grid_s) = cp.modf(frac)
    
    w= cp.sum(cp.stack(((1-frac),frac),axis=2)*kb_2[grid_s.astype(int),:],axis=2)
    return w

@jit(nopython=True,parallel=True)
def grid_prep(w,rind, k_loc, kernel_ind, width, kb_table):
    # gridding preparation
    # output:
    #  w: weights
    #  rind: recon index
    # input:
    #  k_loc: traj
    #  kernel_ind: kernel index
    #  kb_table:
    
    N, k_len = w.shape
    K = kb_table.shape[0]
    ## change the width scale ##
    scale = (width+.5)/(K-1)
    for i in range(k_len):
        t_ind = kernel_ind[0,i]
        for k in range(N):
            rind[k,i] = np.round(k_loc[k,0]+t_ind)
            wid_t = k_loc[k,0] - rind[k,i] 
            w[k,i] = kb_table[int(abs(wid_t/scale))]
            
    return True
            
            
        
def grid_weight_calc(kx, ky, kz, grid_r, shape_stride, width):
    # kx,ky,kz: [N,1],non-scaled trajectory
    
    kb  = kb_table2
    batch_size = kx.shape[0]
    kernel_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernel_ind = kernel_ind[None,:]
    k_len = kernel_ind.size
    
    rind_x = np.zeros((batch_size,k_len))
    rind_y = np.zeros((batch_size,k_len))
    rind_z = np.zeros((batch_size,k_len))
    
    wx = np.zeros((batch_size,k_len))
    wy = np.zeros((batch_size,k_len))
    wz = np.zeros((batch_size,k_len))
    tflag =  grid_prep(wx,rind_x, kx, kernel_ind, width, kb_table)
    tflag =  grid_prep(wy,rind_y, ky, kernel_ind, width, kb_table)
    tflag =  grid_prep(wz,rind_z, kz, kernel_ind, width, kb_table)

    aind_x = (shape_stride[0]*(np.minimum(np.maximum(rind_x,grid_r[0,0]),grid_r[0,1]-1) - grid_r[0,0])).astype(np.int32)
    aind_y = (shape_stride[1]*(np.minimum(np.maximum(rind_y,grid_r[1,0]),grid_r[1,1]-1) - grid_r[1,0])).astype(np.int32)
    aind_z = (shape_stride[2]*(np.minimum(np.maximum(rind_z,grid_r[2,0]),grid_r[2,1]-1) - grid_r[2,0])).astype(np.int32)
    #
    strides_ind = np.zeros((batch_size,k_len,k_len,k_len),dtype=np.int32)
    strides_ind = nu1.badd3(strides_ind,aind_x,aind_y,aind_z)
    w = np.zeros((batch_size,k_len,k_len,k_len))
    w = nu1.btimes3(w,wx,wy,wz)
    w = np.reshape(w,[batch_size,-1])
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
    kernel_ind = cp.arange(np.ceil(-width),np.floor(width)+1)
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
            batch_ind2 = cp.arange(i*batch_size,np.minimum((i+1)*batch_size,samples))
            # load data into GPU from host

            kx = cp.asarray(traj[0,batch_ind,:,0,nP])
            ky = cp.asarray(traj[1,batch_ind,:,0,nP])
            kz = cp.asarray(traj[2,batch_ind,:,0,nP])

            rind_x = cp.rint(kx+kernel_ind).astype(cp.int32)
            rind_y = cp.rint(ky+kernel_ind).astype(cp.int32)
            rind_z = cp.rint(kz+kernel_ind).astype(cp.int32)

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
            # print('Grid time:',time.time()-t0)
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
    kernel_ind = cp.arange(np.ceil(-width),np.floor(width)+1)
    kernel_ind = kernel_ind[None,:]
    k_len = kernel_ind.size
    
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

            rind_x = cp.rint(kx+kernel_ind).astype(cp.int32)
            rind_y = cp.rint(ky+kernel_ind).astype(cp.int32)
            rind_z = cp.rint(kz+kernel_ind).astype(cp.int32)

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

#def gridH2(cdata, kx, ky, kz, ndata, width, shape_stride, grid_r):
    # input:
    #  kx ,ky ,kz: [N, 1]
    #  ndata: [N, 1, nPa]
    #  width: 1
    #  shape_stride: [1,2*width+1]
    #  grid_r: [3,2]
    # output:
    #  cdata: [x, y ,z, nPa]

@cuda.jit()
def gridH2_gpu(c_data1r, c_data1i, n_data1r, n_data1i, traj1, grid_r, width, scale, kb_t):
    # Output:
    #  c_data1: [x,y,z,nPa]
    # Input:
    #  n_data1: [1,N,nPa]
    #  traj1: [3,N]
    #  grid_r: [3,2]
    #  width: 1
    
    Nx, Ny, Nz, nPa = c_data1r.shape
    
    mx = grid_r[0,0]
    my = grid_r[1,0]
    mz = grid_r[2,0]
    N = traj1.shape[1]
    n,npa = cuda.grid(2)
    x = 0
    if n < N :
        xL = int(max(traj1[0,n] - mx + 0.5 - width,0))
        xR = int(min(traj1[0,n] - mx + 0.5 + width,Nx-1))
        yL = int(max(traj1[1,n] - my + 0.5 - width,0))
        yR = int(min(traj1[1,n] - my + 0.5 + width,Ny-1))
        zL = int(max(traj1[2,n] - mz + 0.5 - width,0))
        zR = int(min(traj1[2,n] - mz + 0.5 + width,Nz-1))
        for idx in range(xL,xR):
            kr_x = int(abs(idx - traj1[0,n] + mx )/scale)
            wx = kb_t[kr_x]
            for idy in range(yL,yR):
                kr_y = int(abs(idy - traj1[1,n] + my )/scale)
                wy = kb_t[kr_y]
                for idz in range(zL,zR):
                    kr_z = int(abs(idz - traj1[2,n] + mz )/scale)
                    wz = kb_t[kr_z]
                    wt = wx*wy*wz

                    if npa < nPa: 
                        cuda.atomic.add(c_data1i,(idx,idy,idz,npa),wt*n_data1i[n,npa])
                        cuda.atomic.add(c_data1r,(idx,idy,idz,npa),wt*n_data1r[n,npa])    
    
@cuda.jit()
def gridH2_gpu_t(c_data1r, c_data1i, n_data1r, n_data1i, traj1, grid_r, width, scale, kb_t):
    # Output:
    #  c_data1: [x,y,z,nPa]
    # Input:
    #  n_data1: [1,N,nPa]
    #  traj1: [3,N]
    #  grid_r: [3,2]
    #  width: 1
    
    Nx, Ny, Nz, nPa = c_data1r.shape
    
    mx = grid_r[0,0]
    my = grid_r[1,0]
    mz = grid_r[2,0]
    N = traj1.shape[1]
    n,npa = cuda.grid(2)
    if n < N :
        for nx in range(-width,width+1):
            xt = (traj1[0,n] + nx - mx + .5)//1
            if xt < 0 or xt >Nx-1:
                wx = 0
                indx =0
            else:
                kr_x = int(abs(xt - traj1[0,n] + mx )/scale)
                wx = kb_t[kr_x]
                indx = int(xt)
            for ny in range(-width,width+1):
                yt = (traj1[1,n] + ny - my  + .5)//1
                if yt < 0 or yt >Ny-1:
                    wy = 0
                    indy =0
                else:
                    kr_y = int(abs(yt - traj1[1,n] + my)/scale)
                    wy = kb_t[kr_y]
                    indy = int(yt)
                for nz in range(-width,width+1):
                    zt = (traj1[2,n] + nz - mz + .5)//1
                    if zt <= 0 or zt >Nz-1:
                        wz = 0
                        indz =0
                    else:
                        kr_z = int(abs(zt - traj1[2,n] + mz)/scale)
                        wz = kb_t[kr_z]
                        indz = int(zt)
                        
                    wt = wx*wy*wz
                    if npa < nPa: 
                        cuda.atomic.add(c_data1i,(indx,indy,indz,npa),wt*n_data1i[n,npa])
                        cuda.atomic.add(c_data1r,(indx,indy,indz,npa),wt*n_data1r[n,npa])
                        
            
    

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
    kernel_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernel_ind = kernel_ind[None,:]
    k_len = kernel_ind.size
    
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

            t1 = time.time()
            w, strides_ind = grid_weight_calc(kx, ky, kz, grid_r, shape_stride, width)
            print('weight time:',time.time()-t1)
            
            # Coil Loop Memory limitation
            t2 = time.time()
            
            data_nt = data_n[0,batch_ind,:,:,nP]
            #data_c[:,:,nP] = nu1.cadd2(data_c[:,:,nP],strides_ind,w,data_nt)
            nu1.cadd2(data_c[:,:,nP],strides_ind,w,data_nt)
            print('Grid time:',time.time()-t2)
    # back to host
    data_c = data_c.reshape(shape_grid+[nCoil,nPhase])
    return data_c

def gridH_gpu1(samples, ndata_shape, cdata_shape, traj, data_n, grid_r, width, batch_size, pattern = None):
    # samples: int(N), num of sample
    # traj: [3,N,1,1,1,nbin],non-scaled trajectory
    # data_n: [1,N,1,nC,1,nbin],noncart data
    # grid_r: [2,3] 3D grid range
    # width: Half length of KB window
    # batch_size: limit memory use
    # 
    # kb  = kb_table2
    
    # preparation
    # nCoil = cdata_shape[3]
    nCoil = cdata_shape[Padim-1]
    nPhase = cdata_shape[Phdim-1]
    
    shape_grid = cdata_shape[0:Sdim]
    shape_stride = [shape_grid[2]*shape_grid[1],shape_grid[2],1]
    kernel_ind = np.arange(np.ceil(-width),np.floor(width)+1)
    kernel_ind = kernel_ind[None,:]
    k_len = kernel_ind.size
    
    # data cartesian
    data_c = np.zeros(shape_grid+[nCoil,nPhase],dtype = dtype_c)
    data_n = np.reshape(data_n,ndata_shape)
    
    scale = (width+.5)/(kb_table.shape[0]-1)
    data_cr = np.zeros(tuple(shape_grid+[nCoil]),dtype = dtype_r)
    data_ci = np.zeros(tuple(shape_grid+[nCoil]),dtype = dtype_r)
    
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
            gridH2_gpu[bpg,tpb](data_cr, data_ci, data_nr, data_ni, traj1, np.array(grid_r), width, scale, kb_table)
        
        data_c[:,:,:,:,nP] = data_cr +1j*data_ci 
    print('Total Griding time:',time.time()-t0)
    #cuda.close()
    # back to host
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
