import numpy as np
import cupy as cp
import numba as nb
# GPU

@nb.jit()
def gridH2_gpu(c_data1r, c_data1i, n_data1r, n_data1i, traj1, grid_r, width, scale, kernel):
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
            kr_x = int(abs(idx - traj1[0,n] + mx )/scale + .5)
            wx = kernel[kr_x]
            for idy in range(yL,yR):
                kr_y = int(abs(idy - traj1[1,n] + my )/scale + .5)
                wy = kernel[kr_y]
                for idz in range(zL,zR):
                    kr_z = int(abs(idz - traj1[2,n] + mz )/scale + .5)
                    wz = kernel[kr_z]
                    wt = wx*wy*wz

                    if npa < nPa: 
                        cuda.atomic.add(c_data1i,(idx,idy,idz,npa),wt*n_data1i[n,npa])
                        cuda.atomic.add(c_data1r,(idx,idy,idz,npa),wt*n_data1r[n,npa])  

@nb.jit()
def grid2_gpu(n_data1r, n_data1i, c_data1r, c_data1i, traj1, grid_r, width, scale, kernel):
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
            kr_x = int(abs(idx - traj1[0,n] + mx )/scale + .5)
            wx = kernel[kr_x]
            for idy in range(yL,yR):
                kr_y = int(abs(idy - traj1[1,n] + my )/scale + .5)
                wy = kernel[kr_y]
                for idz in range(zL,zR):
                    kr_z = int(abs(idz - traj1[2,n] + mz )/scale + .5)
                    wz = kernel[kr_z]
                    wt = wx*wy*wz

                    if npa < nPa: 
                        cuda.atomic.add(n_data1i,(n,npa),c_data1i[idx,idy,idz,npa])
                        cuda.atomic.add(n_data1r,(n,npa),c_data1r[idx,idy,idz,npa])  

                        

# modified from sigpy
lin_interp_cuda = '''
    __device__ inline S lin_interp(S* table, int n, S x) {
        if (x >= 1)
           return 0;
        const int idx = x * n;
        const S frac = x * n - idx;
        
        const S left = table[idx];
        S right = 0;
        if (idx != n - 1)
            right = table[idx + 1];
        return (1 - frac) * left + frac * right;
    }
    '''
pos_mod_cuda = '''
    __device__ inline int pos_mod(int x, int n) {
        return (x % n + n) % n;
    }
    '''
gridH2_gput = cp.ElementwiseKernel(
    'raw T output, raw T input, raw S coord, raw S width, raw S table',
    '',
    '''
    const int nPa = output.shape()[3];
    const int nx = output.shape()[0];
    const int ny = output.shape()[1];
    const int nz = output.shape()[2];
    const int coordz_idx[] = {0, i};
    const S posz = coord[coordz_idx];
    const int coordy_idx[] = {1, i};
    const S posy = coord[coordy_idx];
    const int coordx_idx[] = {2, i};
    const S posx = coord[coordx_idx];
    const int startx = ceil(posx - width / 2.0);
    const int starty = ceil(posy - width / 2.0);
    const int startz = ceil(posz - width / 2.0);
    const int endx = floor(posx + width / 2.0);
    const int endy = floor(posy + width / 2.0);
    const int endz = floor(posz + width / 2.0);
    for (int z = startz; z < endz + 1; z++) {
        const S wz = lin_interp(&table[0], table.size(), fabsf((S) z - posz) / (width / 2.0));
        for (int y = starty; y < endy + 1; y++) {
            const S wy = wz * lin_interp(&table[0], table.size(), fabsf((S) y - posy) / (width / 2.0));
            for (int x = startx; x < endx + 1; x++) {
                const S w = wy * lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));
                for (int nP = 0; nP < nPa; nP++) {
                    const int input_idx[] = {i, nP};
                    const int output_idx[] = {pos_mod(x, nx), pos_mod(y, ny), pos_mod(z, nz),nP};

                    const T v = (T) w * input[input_idx];
                    atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])), v.real());
                    atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])) + 1, v.imag());
                }
            }
        }
    }
    ''',
    name='griddingH3_complex',
    preamble=lin_interp_cuda + pos_mod_cuda,
    reduce_dims=False)

grid2_gput = cp.ElementwiseKernel(
    'raw T output, raw T input, raw S coord, raw S width, raw S table',
    '',
    '''
    const int nPa = input.shape()[3];
    const int nx = input.shape()[0];
    const int ny = input.shape()[1];
    const int nz = input.shape()[2];
    const int coordz_idx[] = {0, i};
    const S posz = coord[coordz_idx];
    const int coordy_idx[] = {1, i};
    const S posy = coord[coordy_idx];
    const int coordx_idx[] = {2, i};
    const S posx = coord[coordx_idx];
    const int startx = ceil(posx - width / 2.0);
    const int starty = ceil(posy - width / 2.0);
    const int startz = ceil(posz - width / 2.0);
    const int endx = floor(posx + width / 2.0);
    const int endy = floor(posy + width / 2.0);
    const int endz = floor(posz + width / 2.0);
    for (int z = startz; z < endz + 1; z++) {
        const S wz = lin_interp(&table[0], table.size(), fabsf((S) z - posz) / (width / 2.0));
        for (int y = starty; y < endy + 1; y++) {
            const S wy = wz * lin_interp(&table[0], table.size(), fabsf((S) y - posy) / (width / 2.0));
            for (int x = startx; x < endx + 1; x++) {
                const S w = wy * lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));
                for (int nP = 0; nP < nPa; nP++) {
                    const int input_idx[] = {pos_mod(x, nx), pos_mod(y, ny), pos_mod(z, nz),nP};
                    const int output_idx[] = {i, nP};
                    
                    const T v = (T) w * input[input_idx];
                    atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])), v.real());
                    atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])) + 1, v.imag());
                }
            }
        }
    }
    ''',
    name='gridding3_complex',
    preamble=lin_interp_cuda + pos_mod_cuda,
    reduce_dims=False)