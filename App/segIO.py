import os
import SimpleITK as sitk
import numpy as np
from IO.imgIO import *


def get_training_dataset(data_path, idir, odir):
    # output:
    #  data_fid
    #      - input [Batch, Nchannel, X, Y(, Z)]
    #      - output [Batch, Nchannel, X, Y(, Z)]

    sub_dir = [data_path+x for x in os.listdir(data_path) if os.path.isdir(data_path+x)]
    dataset = []
    for sub in sub_dir:
        dataset.append([read_dcms(sub+idir),read_imgs(sub+odir)])        
    return dataset

def dataset_to_array(dataset,Osize):
    # Osize:[X, Y(, Z)]
    iset = []
    oset = []
    for nset in dataset:
        Isize = list(nset[0].shape)
        dlsize = [(isize-osize)//2 for isize,osize in zip(Isize,Osize)]
        drsize = [(isize+osize) for isize,osize in zip(Osize,dlsize)]
        iset.append(nset[0][dlsize[0]:drsize[0],dlsize[1]:drsize[1],dlsize[2]:drsize[2]])
        oset.append(nset[1][dlsize[0]:drsize[0],dlsize[1]:drsize[1],dlsize[2]:drsize[2]])
    return [np.asarray(iset),np.asarray(oset)]

def prep_for_training2d(dataset, nslice = 3, train_case = 9):
    sub_order = np.random.permutation(len(dataset[0]))
    train_set = {'in_set':[],'out_set':[]}
    val_set = {'in_set':[],'out_set':[]}
    total = 0
    slice_s = nslice//2
    print(sub_order)
    for i in sub_order:
        iset = []
        oset = []
        idata = dataset[0][i]
        odata = dataset[1][i]
        for i in range(idata.shape[0]):
            if i > slice_s and i < idata.shape[0] - slice_s :
                iset.append(idata[i-slice_s:i+slice_s+1,:,:])
                oset.append(odata[i:i+1,:,:])
        
        if total < train_case:
            train_set['in_set'].extend(iset)
            train_set['out_set'].extend(oset)
        else:
            val_set['in_set'].extend(iset)
            val_set['out_set'].extend(oset)
        total = total+1
        
    return train_set,val_set
