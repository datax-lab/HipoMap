import random

import cv2
import numpy as np

from hipomap.WSI_Preprocessing.Preprocessing.Utilities import stainremover_small_patch_remover1


def random_patchcs_with_uniform_distrubutions(img, tile_size):
    img_shape = img.shape
    x_point = random.randint(0, img_shape[0])
    y_point = random.randint(0, img_shape[1])
    try:
        cropped_image = img[x_point:x_point + int(tile_size[0]), y_point:y_point + int(tile_size[1])]
        cleaned_image = stainremover_small_patch_remover1(cropped_image, tile_size)
    except:
        cleaned_image = None
    return cleaned_image


def reconstructionofwholeslide(slide1, new_coordinate_list):
    out = np.zeros_like(slide1)
    for i in range():
        mask = np.zeros((slide1.shape[0], slide1.shape[1]))
        cv2.fillConvexPoly(mask, np.array(new_coordinate_list[i]), 1)
        mask = mask.astype(np.bool)
        out[mask] = slide1[mask]
    out = np.where(out != [0, 0, 0], out, [255, 255, 255])
    return np.array(out)


def all_patches_extraction(slide1, outpath, patch_size):
    reconstrcutedimage = np.zeros_like(slide1)
    for i in range(int(len(slide1[0]) / patch_size[1])):
        for j in range(int(len(slide1) / patch_size[0])):
            sample_img = slide1[j * patch_size[0]:j * patch_size[0] + patch_size[0],
                         i * patch_size[1]:i * patch_size[1] + patch_size[1]]
            patchs = stainremover_small_patch_remover1(sample_img, patch_size)
            if patchs is None:
                None
            else:
                cv2.imwrite('%s/%s_%s.png' % (outpath, i, j), patchs)
                outputImage = cv2.copyMakeBorder(
                    patchs, 5, 5, 5, 5,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 255)
                )
                reconstrcutedimage[j * patch_size[0]:j * patch_size[0] + patch_size[0] + 10,
                i * patch_size[1]:i * patch_size[1] + patch_size[1] + 10] = outputImage
    reconstrcutedimage = np.where(reconstrcutedimage != [0, 0, 0], reconstrcutedimage, [255, 255, 255])
    return reconstrcutedimage


def patch_extraction_random(img, outpath, patch_size, num_of_patches):
    len_list = []
    i = 0
    while len(len_list) < num_of_patches:
        patchs = random_patchcs_with_uniform_distrubutions(img, patch_size)
        if patchs is None:
            None
        else:
            cv2.imwrite('%s/%s.png' % (outpath, i), patchs)
            len_list.append(i)
        i = i + 1
    return
