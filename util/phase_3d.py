import numpy as np
from util import process_3d
from FFT import fft

def lap_unwrap(phase_array, axes = (0,1,2), pad_size = [6,6,6], res = [1,1,1]):
    pad_size = [6,6,6]
    raw_phase = process_3d.zpad(phase_array,pad_size)
    field_of_view = raw_phase.shape[0:3]/np.array(res)
    A = fft.fft(shape=1,axes=axes)

    yy, xx, zz = np.meshgrid(np.arange(0, raw_phase.shape[1]),
                             np.arange(0, raw_phase.shape[0]),
                             np.arange(0, raw_phase.shape[2]))
    xx, yy, zz = ((xx - np.round((raw_phase.shape[0])/2)) / field_of_view[0],
                  (yy - np.round((raw_phase.shape[1])/2)) / field_of_view[1],
                  (zz - np.round((raw_phase.shape[2])/2)) / field_of_view[2])
    k2 = np.square(xx) + np.square(yy) + np.square(zz)+ np.spacing(1)
    k2 = np.expand_dims(k2, axis=3)
    del xx,yy,zz
    
    laplacian = np.zeros(raw_phase.shape, dtype=np.complex)
    phi = np.zeros(raw_phase.shape, dtype=np.complex)
    laplacian = np.cos(raw_phase) * A.IFT(k2*A.FT(np.sin(raw_phase)))-np.sin(raw_phase) * A.IFT(k2*A.FT(np.cos(raw_phase)))
    phi_k = A.FT(laplacian)/k2
    
    if phi.ndim>3 :
        phi_k[np.round((raw_phase.shape[0])/2),np.round((raw_phase.shape[1])/2), np.round((raw_phase.shape[2])/2),...]=0;
    else :
        phi_k[np.round((raw_phase.shape[0])/2),np.round((raw_phase.shape[1])/2), np.round((raw_phase.shape[2])/2)]=0;
    phi = A.IFT(phi_k)
    
    # laplacian = laplacian[pad_size[0]:-pad_size[0],pad_size[1]:-pad_size[1],pad_size[2]:-pad_size[2]]
    if phi.ndim>3 :
        phi = phi[pad_size[0]:-pad_size[0],pad_size[1]:-pad_size[1],pad_size[2]:-pad_size[2],...]
    else :
        phi = phi[pad_size[0]:-pad_size[0],pad_size[1]:-pad_size[1],pad_size[2]:-pad_size[2]]
    phi = np.real(phi)
    return phi