import cv2

from hipomap.WSI_Preprocessing.Preprocessing.Denoising import de_noising
from hipomap.WSI_Preprocessing.Preprocessing.Patch_extraction_creatia import patch_extraction_random, \
    all_patches_extraction


def extractingPatches(inputsvs, outputpath, magnification, patch_extraction_creatia=None, num_of_patches=2000,
                      filtering="GaussianBlur", patch_size=(256, 256), upperlimit=900, lowerlimit=300,
                      red_value=(80, 220), green_value=(80, 200), blue_value=(80, 170), reconstructedimagepath=None,
                      Annotation=None, Annotatedlevel=0, Requiredlevel=0, img=None, outpath=None):
    slide = de_noising(inputsvs, magnification, filtering, patch_size, upperlimit, lowerlimit, red_value, green_value,
                       blue_value, Annotation, Annotatedlevel, Requiredlevel)
    if patch_extraction_creatia == None:
        print("patch_extraction_started")
        reconstructedimage = all_patches_extraction(slide, outputpath, patch_size)
        print("patch_extraction_done")
        if reconstructedimagepath != None:
            cv2.imwrite("%s/%s.png" % (reconstructedimagepath, inputsvs.split("/")[-1][:-4]), reconstructedimage)
    else:
        patch_extraction_random(img, outpath, patch_size, num_of_patches)
    print("reconstructing image")
    print("done")
    return reconstructedimage
