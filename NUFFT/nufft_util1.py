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

@jit(nopython=True,parallel=True)
def grid2(n_data1, c_data1, traj1, grid_r, width, scale, kernel):
    # Output:
    #  c_data1: [x,y,z,nPa]
    # Input:
    #  n_data1: [1,N,nPa]
    #  traj1: [3,N]
    #  grid_r: [3,2]
    #  width: 1
    
    Nx, Ny, Nz, nPa = c_data1.shape
    
    mx = grid_r[0,0]
    my = grid_r[1,0]
    mz = grid_r[2,0]
    N = traj1.shape[1]
    x = 0
    for n in range(N):
        xL = int(max(traj1[0,n] - mx + 0.5 - width,0))
        xR = int(min(traj1[0,n] - mx + 0.5 + width,Nx-1))
        yL = int(max(traj1[1,n] - my + 0.5 - width,0))
        yR = int(min(traj1[1,n] - my + 0.5 + width,Ny-1))
        zL = int(max(traj1[2,n] - mz + 0.5 - width,0))
        zR = int(min(traj1[2,n] - mz + 0.5 + width,Nz-1))
        
        for idx in range(xL,xR):
            kr_x = int(abs(idx - traj1[0,n] + mx )/scale)
            wx = kernel[kr_x]
            for idy in range(yL,yR):
                kr_y = int(abs(idy - traj1[1,n] + my )/scale)
                wy = kernel[kr_y]
                for idz in range(zL,zR):
                    kr_z = int(abs(idz - traj1[2,n] + mz )/scale)
                    wz = kernel[kr_z]
                    wt = wx*wy*wz
                    for npa in range(nPa): 
                        n_data1[n,npa] += wt*c_data1[idx,idy,idz,npa] 

@jit(nopython=True,parallel=True)
def gridH2(c_data1, n_data1, traj1, grid_r, width, scale, kernel):
    # Output:
    #  c_data1: [x,y,z,nPa]
    # Input:
    #  n_data1: [1,N,nPa]
    #  traj1: [3,N]
    #  grid_r: [3,2]
    #  width: 1
    
    Nx, Ny, Nz, nPa = c_data1.shape
    
    mx = grid_r[0,0]
    my = grid_r[1,0]
    mz = grid_r[2,0]
    N = traj1.shape[1]
    x = 0
    for n in range(N):
        xL = int(max(traj1[0,n] - mx + 0.5 - width,0))
        xR = int(min(traj1[0,n] - mx + 0.5 + width,Nx-1))
        yL = int(max(traj1[1,n] - my + 0.5 - width,0))
        yR = int(min(traj1[1,n] - my + 0.5 + width,Ny-1))
        zL = int(max(traj1[2,n] - mz + 0.5 - width,0))
        zR = int(min(traj1[2,n] - mz + 0.5 + width,Nz-1))
        
        for idx in range(xL,xR):
            kr_x = int(abs(idx - traj1[0,n] + mx )/scale)
            wx = kernel[kr_x]
            for idy in range(yL,yR):
                kr_y = int(abs(idy - traj1[1,n] + my )/scale)
                wy = kernel[kr_y]
                for idz in range(zL,zR):
                    kr_z = int(abs(idz - traj1[2,n] + mz )/scale)
                    wz = kernel[kr_z]
                    wt = wx*wy*wz
                    for npa in range(nPa): 
                        c_data1[idx,idy,idz,npa] += wt*n_data1[n,npa]
                        

                                              