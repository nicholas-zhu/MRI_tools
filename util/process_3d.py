import numpy as np

def zpad(img_arr, pad_size=[0,0,0], mode='constant'):
    p_size = pad_size
    if img_arr.ndim>3:
        for i in range(3,img_arr.ndim):
            p_size = p_size + [0]    
    
    p_size = tuple([(size,size) for size in p_size])
    img_arr_n = np.pad(img_arr, p_size, mode)
    return img_arr_n

def zpad2(img_arr, r_size, mode='constant'):
    img_size = np.array(img_arr.shape)
    print(img_arr.shape, r_size)
    p_sizeL = np.round((np.array(r_size)-img_size)/2)
    p_sizeR = np.floor((np.array(r_size)-img_size)/2)
    p_size  = np.stack([p_sizeL,p_sizeR],axis=1) 
    p_size = tuple(map(tuple,p_size.astype(np.int32)))
    
    img_arr_n = np.pad(img_arr, p_size, mode)
    return img_arr_n
    
def crop(img_arr, crop_size):
    if img_arr.ndim > 3:
        img_arr_c = img_arr[crop_size[0]:-crop_size[0],crop_size[1]:-crop_size[1],crop_size[2]:-crop_size[2],...]
    else:
        img_arr_c = img_arr[crop_size[0]:-crop_size[0],crop_size[1]:-crop_size[1],crop_size[2]:-crop_size[2]]
    return img_arr_c