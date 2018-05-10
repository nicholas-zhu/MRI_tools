import numpy as np
from numba import jit

try:
    import cupy as cp
    from numba import cuda
    CUDA_flag = True
except ImportError:
    CUDA_flag = False

@jit(nopython=True,parallel=True)
def cadd(obj_arr, ind_arr, sub_arr):
    # o[ind_arr] += s1
    # obj_arr: [No,nPa]
    # ind_arr: [Ns,1]
    # sub_arr: [Ns,nPa]
    No, nPa = obj_arr.shape
    Ns, nPa1 = ind_arr.shape
    
    for i in range(Ns):
        tind = ind_arr[i,0]
        for j in range(nPa):
            obj_arr[tind,j] += sub_arr[i,j]
            
    return obj_arr

@jit(nopython=True,parallel=True)
def cadd2(obj_arr, ind_arr, sub_arr1, sub_arr2):
    # o[ind_arr] += s1*s2
    # obj_arr: [No,nPa]
    # ind_arr: [Ns,Nw]
    # sub_arr1: [Ns,Nw]
    # sub_arr2: [Ns,1,nPa]
    
    No, nPa = obj_arr.shape
    Ns, Nw = ind_arr.shape
    
    for i in range(Ns):
        for j in range(Nw):
            tind = ind_arr[i,j]
            tsub_arr1 = sub_arr1[i,j] 
            for k in range(nPa):
                obj_arr[tind,k] += (tsub_arr1 * sub_arr2[i,0,k])
            
    return obj_arr

@jit(nopython=True,parallel=True)
def cadd3(obj_arr, ind_arr, sub_arr1, sub_arr2):
    # o += s1*s2[ind_arr]
    # obj_arr: [No,nPa]
    # ind_arr: [No,Nw]
    # sub_arr1: [No,Nw]
    # sub_arr2: [Ns,1,nPa]
    
    No, nPa = obj_arr.shape
    Nw = sub_arr1.shape[1]
    Ns = sub_arr2.shape[0]
    
    for i in range(No):
        for j in range(Nw):
            tind = ind_arr[i,j]
            tsub_arr1 = sub_arr1[i,j] 
            for k in range(nPa):
                obj_arr[i,k] += (tsub_arr1 * sub_arr2[tind,0,k])
            
    return obj_arr

@jit(nopython=True,parallel=True)
def badd3(obj_arr, sub_arr1, sub_arr2, sub_arr3):
    # 3d broadcast add
    # o += s1+s2+s3
    # obj_arr: [No,nx,ny,nz]
    # sub_arr1: [No,nx]
    # sub_arr2: [No,ny]
    # sub_arr3: [Ns,nz]
    
    No,nx,ny,nz = obj_arr.shape
    for n in range(No):
        for i in range(nx):
            sx = sub_arr1[n,i]
            for j in range(ny):
                sy = sub_arr2[n,j]
                for k in range(nz):
                    sz = sub_arr3[n,k] 
                    obj_arr[n,i,j,k] = sx+sy+sz
            
    return obj_arr

@jit(nopython=True,parallel=True)
def btimes3(obj_arr, sub_arr1, sub_arr2, sub_arr3):
    # 3d broadcast times
    # o += s1+s2+s3
    # obj_arr: [No,nx,ny,nz]
    # sub_arr1: [No,nx]
    # sub_arr2: [No,ny]
    # sub_arr3: [Ns,nz]
    
    No,nx,ny,nz = obj_arr.shape
    for n in range(No):
        for i in range(nx):
            sx = sub_arr1[n,i]
            for j in range(ny):
                sy = sub_arr2[n,j]
                for k in range(nz):
                    sz = sub_arr3[n,k] 
                    obj_arr[n,i,j,k] = sx*sy*sz
            
    return obj_arr