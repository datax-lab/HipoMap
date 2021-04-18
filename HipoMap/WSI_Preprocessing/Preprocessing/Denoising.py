import cv2
import math
import numpy as np
import os
from HipoMap.WSI_Preprocessing.Preprocessing.WSI_Scanning import readWSI
from HipoMap.WSI_Preprocessing.Preprocessing.Utilities import garbage_collector,denoising_lowermiginification_guassianblur,denoising_RGB_Thersholding,denoising_No_filters,dictionary,denoising_using_GaussianBlur


def removing_small_tissues(img, meannumber):
    cv2.imwrite("tempt.png",cv2.cvtColor(np.array(img, dtype = "uint8"), cv2.COLOR_RGB2BGR))
    img_n = cv2.imread("tempt.png",0)
    kernalnew = np.ones((3, 3), np.uint8)
    kernalnew1 = np.ones((1, 1), np.uint8)
    img_d = cv2.erode(img_n, kernalnew, iterations=10)
    img_d = cv2.dilate(img_d, kernalnew1, iterations=5)

    binary_map1 = (img_d < 200).astype(np.uint8)

    X = cv2.connectedComponentsWithStats(binary_map1, 8)[2][1:]
    output = cv2.connectedComponentsWithStats(binary_map1, 8)[1]
    img2 = np.zeros((output.shape))

    for i in range(len(X)):
        if X[i,4]>X[:,4].mean()*int(meannumber):
            img2[output == i + 1] = 255
            cv2.imwrite("exampleee2.png", img2*255)
            

    img3 = cv2.imread("exampleee2.png")
    img3 = img3/255
    img4 = img3*img
    img4[np.where((img4 == [0,0,0]).all(axis = 2))] = [255,255,255]
    return img4

def denoising(inputsvs,magnification,filtering = "GaussianBlur",patch_size = (256,256),upperlimit = 900, lowerlimit = 300, red_value = (80,220), green_value = (80,200), blue_value = (80, 170), Annotation = None, Annotatedlevel = 0, Requiredlevel = 0, meannumber = 8):
    img,slide_dimensions= readWSI(inputsvs,magnification,Annotation, Annotatedlevel, Requiredlevel)
    dictx = dictionary(slide_dimensions)
    if filtering == "GaussianBlur":
        img_n = denoising_using_GaussianBlur(inputsvs,magnification,img,dictx,patch_size ,upperlimit, lowerlimit,Annotation , Annotatedlevel , Requiredlevel)
        out = removing_small_tissues(img_n, meannumber)
        
    elif filtering == "RGB":
        mask = denoising_RGB_Thersholding(img,slide_dimensions, magnification,dictx,patch_size,red_value,green_value,blue_value)
        out = np.zeros_like(img)
        print("cleaning image at high mignification")
        mask = mask.astype(np.bool)
        out[mask] = img[mask]
        out = np.where(out != [0, 0, 0], out, [255,255,255])
        print("cleaning WSI done")
    #     cv2.imwrite("/home/pagenet2/PageNet2/Data Preprocessing Pipeline/WSI_Precessing_test/cleanedimages/%s/cleanedsmallf.png"%(inputsvs.split("/")[-1][:-4]),out)
        garbage_collector()
        print("exisiting cleaning")
        
    else:
        mask = denoising_No_filters(img,slide_dimensions, magnification,dictx)
        out = np.zeros_like(img)
        print("cleaning image at high mignification")
        mask = mask.astype(np.bool)
        out[mask] = img[mask]
        out = np.where(out != [0, 0, 0], out, [255,255,255])
        print("cleaning WSI done")
    #     cv2.imwrite("/home/pagenet2/PageNet2/Data Preprocessing Pipeline/WSI_Precessing_test/cleanedimages/%s/cleanedsmallf.png"%(inputsvs.split("/")[-1][:-4]),out)
        garbage_collector()
        print("exisiting cleaning")
    return out
