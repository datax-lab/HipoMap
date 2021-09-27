import cv2
import numpy as np
from openslide import (OpenSlide)

from hipomap.WSI_Preprocessing.Preprocessing.Annotation_parsing import extracting_roi_annotations


def readWSI(slide_path, magnification, Annotation, Annotatedlevel, Requiredlevel):
    slide = OpenSlide(slide_path)
    slide_dimensions = slide.level_dimensions

    if len(slide_dimensions) == 3:

        dictx = {"20x": 0, "10x": 1, "5x": 2}

        if magnification == "40x":
            raise ValueError("This image doesnot have 40x maginification")

    #         if magnification != dictx.keys():
    #             raise Exception("maginification should be 40x, 20x, 10x 0r 5x")

    else:

        dictx = {"40x": 0, "20x": 1, "10x": 2, "5x": 3}
    #     if magnification != dictx.keys():
    #         raise Exception("maginification should be 40x, 20x, 10x 0r 5x")

    print(dictx[magnification])
    mag = dictx[magnification]
    slide_img_1 = slide.read_region((0, 0), mag,
                                    (slide.level_dimensions[mag][0], slide.level_dimensions[mag][1])).convert('RGB')
    #     print(slide.level_dimensions[mag][0], slide.level_dimensions[mag][1])
    slide_img_1 = np.asarray(slide_img_1, dtype="int32")
    cv2.imwrite("20x.png", slide_img_1)
    #     print(Annotation)
    if Annotation != None:
        #         print('yes')
        new_cordinate_list = extracting_roi_annotations(Annotation, slide_dimensions, Annotatedlevel, Requiredlevel)
        #         print(new_cordinate_list)
        slide_img_2 = reading_WSI_with_annotations(slide_img_1, new_cordinate_list)

        cv2.imwrite("20x1.png", slide_img_2)
        return slide_img_2, slide_dimensions
    else:
        #         print('no')
        return slide_img_1, slide_dimensions


def reading_WSI_with_annotations(slide_img, new_cordinate_list):
    slide1 = slide_img
    out = np.zeros_like(slide1)
    for i in range(len(new_cordinate_list)):
        mask = np.zeros((slide1.shape[0], slide1.shape[1]))
        #         print((slide1.shape[0], slide1.shape[1]))
        #         print(new_cordinate_list[i])
        cv2.fillConvexPoly(mask, np.array(new_cordinate_list[i]), 1)
        cv2.imwrite("example%s.png" % i, mask * 255)
        mask = mask.astype(np.bool)
        out[mask] = slide1[mask]
        # out[out == [0,0,0]] =  [255,255,255]
    #         cv2.imwrite("example%s.png"%i,mask)
    out[np.where((out == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    # print('yes')
    cv2.imwrite("out.png", out)
    return np.array(out)
