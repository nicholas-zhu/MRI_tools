import numpy as np
import pyfftw as pfft
import multiprocessing
try:
    import cupy as cp
    CUDA_flag = True
except ImportError:
    CUDA_flag = False


N_max = multiprocessing.cpu_count()
N_min = 1
N = N_max//2
N_threads = np.maximum(N,N_min)

class fft:
    def __init__(self, shape, axes = (0,1,2)):
        self.shape = shape
        self.axes = axes
    
    def FT(self,a):
        b = pfft.interfaces.numpy_fft.ifftshift(a,axes=self.axes)
        b = pfft.interfaces.numpy_fft.fftn(b,axes=self.axes,threads = N_threads)
        b = pfft.interfaces.numpy_fft.fftshift(b,axes=self.axes)
        return b
    
    def IFT(self,a):
        b = pfft.interfaces.numpy_fft.ifftshift(a,axes=self.axes)
        b = pfft.interfaces.numpy_fft.ifftn(b, axes=self.axes,threads = N_threads)
        b = pfft.interfaces.numpy_fft.fftshift(b, axes=self.axes)
        return b
    
class fft_gpu:
    def __init__(self, axes = (0,1,2),seg = 4):
        self.axes = axes
        self.seg = seg
    
    def FT(self,a):
        I_shape = list(a.shape)
        b = np.fft.ifftshift(a,axes=self.axes).reshape(tuple(I_shape[0:3]+[-1]))
        print(b.shape)
        N = b.shape[3]
        for i in range((N-1)//self.seg + 1):
            ind = np.arange(i*self.seg,np.minimum((i+1)*self.seg,N))
            b_gpu = cp.asarray(b[:,:,:,ind])
            b_gpu = cp.fft.fftn(b_gpu,axes = (0,1,2),norm='ortho')
            b[:,:,:,ind] = cp.asnumpy(b_gpu)
        b = np.fft.fftshift(b,axes=self.axes).reshape(I_shape)
        return b
    
    def IFT(self,a):
        I_shape = list(a.shape)
        b = np.fft.ifftshift(a,axes=self.axes).reshape(tuple(I_shape[0:3]+[-1]))
        print(b.shape)
        N = b.shape[3]
        for i in range((N-1)//self.seg + 1):
            ind = np.arange(i*self.seg,np.minimum((i+1)*self.seg,N))
            b_gpu = cp.asarray(b[:,:,:,ind])
            b_gpu = cp.fft.ifftn(b_gpu,axes = (0,1,2),norm='ortho')
            b[:,:,:,ind] = cp.asnumpy(b_gpu)
        b = np.fft.fftshift(b,axes=self.axes).reshape(I_shape)
        return b