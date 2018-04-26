from __future__ import print_function
 
import SimpleITK as sitk
import sys, os

def read_dcms(dicom_folder):
    print( "Reading Dicom directory:", dicom_folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image = sitk.GetArrayFromImage(reader.Execute())
    return image

def read_dcm(dicom_file):
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_file)
    image = sitk.GetArrayFromImage(reader.Execute())
    
    return image