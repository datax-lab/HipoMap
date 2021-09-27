import cv2
import numpy as np
from openslide import (OpenSlide)


def relu(x):
    return x * (x > 0)


def read_wsi(slide_path, magnification):
    slide = OpenSlide(slide_path)
    slide_dimensions = slide.level_dimensions

    if len(slide_dimensions) == 3:

        dictx = {"20x": 0, "10x": 1, "5x": 2}

        if magnification == "40x":
            raise ValueError("This image does not have 40x maginification")

    else:
        dictx = {"40x": 0, "20x": 1, "10x": 2, "5x": 3}

    print(dictx[magnification])
    mag = dictx[magnification]
    slide_img_1 = slide.read_region((0, 0), mag,
                                    (slide.level_dimensions[mag][0], slide.level_dimensions[mag][1])).convert('RGB')
    slide_img_1 = np.asarray(slide_img_1, dtype="int32")
    cv2.imwrite("20x.png", slide_img_1)

    return slide_img_1, slide_dimensions


def stain_remover_small_patch_remover(img, patch_size):
    if len(img) < patch_size[1] or len(img[0]) < patch_size[0]:
        return None
    else:
        Xb = []
        Xg = []
        Xr = []
        for i in range(len(img)):
            Xb.append(np.mean(img[i][:, 0]))
            Xg.append(np.mean(img[i][:, 1]))
            Xr.append(np.mean(img[i][:, 2]))

        if np.mean(Xr) < 0 or np.mean(Xr) > 255:
            return None
        elif np.mean(Xg) < 0 or np.mean(Xg) > 255:
            return None
        elif np.mean(Xb) < 0 or np.mean(Xb) > 255:
            return None
        else:
            cv2.imwrite("tempN1.png", img)
            img_bg = cv2.imread("tempN1.png", 0)
            if (img_bg.mean() < 230) and (img_bg.mean() > 20):
                return img
            else:
                return None
