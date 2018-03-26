from __future__ import print_function
 
import SimpleITK as sitk
import sys, os

def read_dcm(dicom_folder):
    print( "Reading Dicom directory:", dicom_folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image
