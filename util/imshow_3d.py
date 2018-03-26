import numpy as np
from matplotlib import pyplot as plt

def imshow3d(image, ncolumn = None, nrow = None):
    image = np.squeeze(image)
    Isize = image.shape
    assert len(Isize)<=3, 'wrong Image shape, should be 3D'
    if len(Isize) is 3:
        if ncolumn is None:
            ncolumn = np.ceil(np.sqrt(Isize[-1]))
        ncolumn = np.maximum(ncolumn,1).astype(np.int32)
        
        if nrow is None:
            nrow = np.ceil(Isize[-1]/ncolumn)
        nrow = np.maximum(nrow,1).astype(np.int32)
        
        nt = nrow * ncolumn
        image_plt = np.zeros([Isize[0],Isize[1],nt]) + np.min(image)
        nc = np.minimum(nt, Isize[-1])
        image_plt[:,:,0:nc] = image[:,:,0:nc]
        
        image_plt2 = np.transpose(np.reshape(image_plt,(Isize[0], Isize[1], nrow, ncolumn)),(0,2,1,3))
        image_plt = np.reshape(image_plt2,(Isize[0]*nrow,Isize[1]*ncolumn),order='F')

    else:
        image_plt = image       
    
    plt.imshow(image_plt)
    plt.ioff()
    plt.colorbar()
    plt.show()