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
    img_size = np.array(img_arr.shape[0:3])
    p_sizeL = np.floor(np.array(r_size)/2)-np.floor(img_size/2)
    p_sizeR = np.ceil(np.array(r_size)/2)-np.ceil(img_size/2)
    p_size  = np.maximum(np.stack([p_sizeL,p_sizeR],axis=1),0).astype(int).tolist()
    if img_arr.ndim>3:
        for i in range(3,img_arr.ndim):
            p_size = p_size+[[0,0]]
    p_size = tuple(map(tuple,p_size))
    img_arr_n = np.pad(img_arr, p_size, mode)
    return img_arr_n
    
def crop(img_arr, crop_size):
    if img_arr.ndim > 3:
        img_arr_c = img_arr[crop_size[0]:-crop_size[0],crop_size[1]:-crop_size[1],crop_size[2]:-crop_size[2],...]
    else:
        img_arr_c = img_arr[crop_size[0]:-crop_size[0],crop_size[1]:-crop_size[1],crop_size[2]:-crop_size[2]]
    return img_arr_c

def crop2(img_arr, r_size):
    img_size = np.array(img_arr.shape[0:3])
    
    p_sizeL = -(np.floor(np.array(r_size)/2)-np.floor(img_size/2)).astype(int)
    p_sizeR = (np.ceil(np.array(r_size)/2)-np.ceil(img_size/2)).astype(int)

    xL, yL, zL = tuple([i if i>0 else None for i in p_sizeL])
    xR, yR, zR = tuple([i if i<0 else None for i in p_sizeR])
   
    if img_arr.ndim>3:
        img_arr_n = img_arr[xL:xR,yL:yR,zL:zR,:]
    else:
        img_arr_n = img_arr[xL:xR,yL:yR,zL:zR]
    return img_arr_n

def resize(img_arr, output_size):
    img_size = np.array(img_arr.shape)
    if img_arr.shape[0:3]==output_size:
        img_arr_n = img_arr
    else:
        img_arr_n = crop2(img_arr, output_size)
        img_arr_n = zpad2(img_arr_n, output_size)

    return img_arr_n
    
    