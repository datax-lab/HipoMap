import numpy as np
from src.Localization import localization_with_roi,localization_with_out_roi
from src.Patch_extraction_creteria import patch_extraction_random,all_patches_extarction
import cv2
from src.WSI_Scanning import reading_WSI,reading_WSI_with_annotations
import gc
import pprint
import os
from src.Utilities import making_one_image,reading_image_at_low_magnification,otsu_s,cleaning_image_at_high_mignification,garbage_collector
from numba import jit, cuda

def Extraction_slides_with_annotations(inputxml,inputsvs,outpath,Patch_extraction_creatia,patch_size,num_of_patches):
    correctedslide = localization_with_roi(inputxml, inputsvs)
    if Patch_extraction_creatia == 'random':
        patch_extraction_random(correctedslide, outpath,patch_size, num_of_patches)
    else:
        all_patches_extarction(correctedslide, outpath, patch_size)
    # patch_extraction(correctedslide, out_put_path)
    return
# @jit(target = "cuda")



def Extraction_slides_without_annotations(inputsvs,outpath,Patch_extraction_creatia,patch_size,num_of_patches):
    os.mkdir("Reconstructedimages")
    slidei = cleaning_image_at_high_mignification(inputsvs)
    if Patch_extraction_creatia == 'random':
        patch_extraction_random(slidei, outpath,patch_size, num_of_patches)
    else:
        print("patchs extraction started")
        reconstrcutedimage = all_patches_extarction(slidei, outpath, patch_size)
        print("patch extraction is completed")
        print("reconstructing image")
        cv2.imwrite("Reconstructedimages/%s.png"%(inputsvs.split("/")[-1][:-4]),reconstrcutedimage)
        print("exiting reconstruction")
    garbage_collector()
    print("Package succesfully extracted for WSI %s"%inputsvs)
    return
