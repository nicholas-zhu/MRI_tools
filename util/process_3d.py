import numpy as np

def zpad(img_arr, pad_size=[0,0,0], mode='constant'):
    p_size = pad_size
    if img_arr.ndim>3:
        for i in range(3,img_arr.ndim):
            p_size = p_size + [0]    
    
    p_size = tuple([(size,size) for size in p_size])
    img_arr_n = np.pad(img_arr, p_size, mode)
    return img_arr_n

