import numpy as np
import cv2
from src.WSI_Scanning import reading_WSI,reading_WSI_with_annotations
from src.Annotation_parsing import extracting_roi_annotations
from src.Utilities import stainremover_small_patch_remover,stainremover_small_patch_remover1

def localization_with_roi(inputxml, inputsvs):
    slide_dimensions, slide_img = reading_WSI(inputsvs)
    new_cordinate_list  = extracting_roi_annotations(inputxml,slide_dimensions)
    new_image = reading_WSI_with_annotations(slide_img, new_cordinate_list)
    return new_image



def localization_with_out_roi(inputsvs):
    slide1 = inputsvs
    patch_x = 20
    for i in range(int((len(slide1[0]))/patch_x)+1):
        for j in range(int((len(slide1))/patch_x)+1):
            sample_img = slide1[j*patch_x:j*patch_x+patch_x,i*patch_x:i*patch_x+patch_x]
            sample_img_new = stainremover_small_patch_remover(sample_img,(patch_x,patch_x),upperlimit = 1300, lowerlimit = 350)
            if sample_img_new is None:
                slide1[j * patch_x:j * patch_x + patch_x, i * patch_x:i * patch_x+ patch_x]  = np.zeros_like(sample_img)
            else:
                slide1[j*patch_x:j*patch_x+patch_x,i*patch_x:i*patch_x+patch_x] = sample_img
    slide1 = np.where(slide1 != [0, 0, 0], slide1, [255, 255, 255])
    return slide1