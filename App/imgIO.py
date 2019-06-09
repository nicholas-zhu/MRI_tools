import os
import SimpleITK as sitk
import numpy as np

def read_dcms(dicom_folder):
    print( "Reading Dicom directory:", dicom_folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image = sitk.GetArrayFromImage(reader.Execute())
    return image

def read_imgs(img_folder):
    print( "Reading Image directory:", img_folder)
    imgs = []
    for img in sorted(os.listdir(img_folder), reverse=True):
        imgs.append(read_img(img_folder+img))
    imgs = np.asarray(imgs)
    return imgs
    
def read_img(img_file):
    reader = sitk.ImageFileReader()
    reader.SetFileName(img_file)
    image = sitk.GetArrayFromImage(reader.Execute())
    
    return image

def load_dcm_dataset(fid_list):
    # Output:
    #  dataarray:[Batch, Channel, X, Y]
    data = []
    for fid in fid_list:
        dcm = read_dcms(fid)
        data.append(dcm)
    return np.asarray(data)

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

