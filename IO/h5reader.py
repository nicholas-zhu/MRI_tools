import h5py
import numpy as np

def uwute_read(h5file_name_base, nTE = 1, return_flag = False):
    fname = h5file_name_base+'.h5'
    
    print('UWUTE Loading data ...')
    with h5py.File(fname,'r') as fr:
        # time = fr['Gating']['time'].ravel()
        # order = np.argsort(time)

        traj = []
        data = []
        dcf = []
        # Noise
        noise = fr['Kdata']['Noise']['real'] + 1j*fr['Kdata']['Noise']['imag']
        nCoil = noise.shape[0]
        
        # data
        for nte in range(nTE):
            time = np.squeeze(fr['Gating']['TIME_E{nte}'.format(nte = nte)])
            order = np.argsort(time)
            trajn = []
            for i in ['Y','X','Z']:
                trajn.append(fr['Kdata']['K{i}_E{nte}'.format(i = i, nte = nte)][0][order])
            trajn = np.stack(trajn,axis=-1)
            traj.append(trajn)
            
            datanc = []
            for nc in range(nCoil):
                d = fr['Kdata']['KData_E{nte}_C{nc}'.format(nte = nte, nc = nc)]
                datanc.append(d['real'][0][order] + 1j*d['imag'][0][order])
            data.append(datanc)
            
            dcf.append(fr['Kdata']['KW_E{nte}'.format(nte = nte)])
        data = np.array(data)
        traj = np.array(traj)  
        dcf = np.array(dcf)
        print(data.shape,traj.shape,dcf.shape)
        data = np.transpose(data[:,:,:,:,None],axes=(4,3,2,1,0))
        traj = np.transpose(traj[:,None,:,:,:]  ,axes=(4,3,2,1,0))
        dcf = np.transpose(dcf[:,:,:,:,None],axes=(4,3,2,1,0))
        
        if return_flag:
            print('Writing data ...')
            np.save(h5file_name_base+'_data',data)
            np.save(h5file_name_base+'_traj',traj)
            np.save(h5file_name_base+'_dcf',dcf)
            np.save(h5file_name_base+'_noise',noise)
            print('Done!')
        else:
            return data,traj,dcf,noise
                       
            