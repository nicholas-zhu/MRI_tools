import numpy as np
import pyfftw as pfft
import multiprocessing

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