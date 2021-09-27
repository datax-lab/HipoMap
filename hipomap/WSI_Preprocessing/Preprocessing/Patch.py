import os

import cv2
from hipomap.WSI_Preprocessing.Preprocessing.Patch_extraction_creatia import patch_extraction_random, all_patches_extraction

from hipomap.WSI_Preprocessing.Preprocessing.Localization import localization_with_roi
from hipomap.WSI_Preprocessing.Preprocessing.Utilities import garbage_collector


def Extraction_slides_with_annotations(inputxml, inputsvs, outpath, Patch_extraction_creatia, patch_size,
                                       num_of_patches):
    correctedslide = localization_with_roi(inputxml, inputsvs)
    if Patch_extraction_creatia == 'random':
        patch_extraction_random(correctedslide, outpath, patch_size, num_of_patches)
    else:
        all_patches_extraction(correctedslide, outpath, patch_size)
    # patch_extraction(correctedslide, out_put_path)
    return


# @jit(target = "cuda")


def Extraction_slides_without_annotations(inputsvs, outpath, Patch_extraction_creatia, patch_size, num_of_patches):
    os.mkdir("Reconstructedimages")
    slidei = cleaning_image_at_high_mignification(inputsvs)
    if Patch_extraction_creatia == 'random':
        patch_extraction_random(slidei, outpath, patch_size, num_of_patches)
    else:
        print("patchs extraction started")
        reconstrcutedimage = all_patches_extarction(slidei, outpath, patch_size)
        print("patch extraction is completed")
        print("reconstructing image")
        cv2.imwrite("Reconstructedimages/%s.png" % (inputsvs.split("/")[-1][:-4]), reconstrcutedimage)
        print("exiting reconstruction")
    garbage_collector()
    print("Package succesfully extracted for WSI %s" % inputsvs)
    return
