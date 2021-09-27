import cv2
import numpy as np

from hipomap.WSI_Preprocessing.Preprocessing.Utilities import garbage_collector, denoising_RGB_Thersholding, \
    denoising_No_filters, dictionary, denoising_using_GaussianBlur
from hipomap.WSI_Preprocessing.Preprocessing.WSI_Scanning import readWSI


def removing_small_tissues(img, mean_number):
    cv2.imwrite("tempt.png", cv2.cvtColor(np.array(img, dtype="uint8"), cv2.COLOR_RGB2BGR))
    img_n = cv2.imread("tempt.png", 0)
    kernel_new = np.ones((3, 3), np.uint8)
    kernel_new1 = np.ones((1, 1), np.uint8)
    img_d = cv2.erode(img_n, kernel_new, iterations=10)
    img_d = cv2.dilate(img_d, kernel_new1, iterations=5)

    binary_map1 = (img_d < 200).astype(np.uint8)

    X = cv2.connectedComponentsWithStats(binary_map1, 8)[2][1:]
    output = cv2.connectedComponentsWithStats(binary_map1, 8)[1]
    img2 = np.zeros((output.shape))

    for i in range(len(X)):
        if X[i, 4] > X[:, 4].mean() * int(mean_number):
            img2[output == i + 1] = 255
            cv2.imwrite("exampleee2.png", img2 * 255)

    img3 = cv2.imread("exampleee2.png")
    img3 = img3 / 255
    img4 = img3 * img
    img4[np.where((img4 == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    return img4


def de_noising(input_svs, magnification, filtering="GaussianBlur", patch_size=(256, 256), upper_limit=900,
               lower_limit=300, red_value=(80, 220), green_value=(80, 200), blue_value=(80, 170), annotation=None,
               annotated_level=0, required_level=0, mean_number=8):
    img, slide_dimensions = readWSI(input_svs, magnification, annotation, annotated_level, required_level)
    dictx = dictionary(slide_dimensions)
    if filtering == "GaussianBlur":
        img_n = denoising_using_GaussianBlur(input_svs, magnification, img, dictx, patch_size, upper_limit, lower_limit,
                                             annotation, annotated_level, required_level)
        out = removing_small_tissues(img_n, mean_number)

    elif filtering == "RGB":
        mask = denoising_RGB_Thersholding(img, slide_dimensions, magnification, dictx, patch_size, red_value,
                                          green_value, blue_value)
        out = np.zeros_like(img)
        print("cleaning image at high mignification")
        mask = mask.astype(np.bool)
        out[mask] = img[mask]
        out = np.where(out != [0, 0, 0], out, [255, 255, 255])
        print("cleaning WSI done")

        garbage_collector()
        print("exisiting cleaning")

    else:
        mask = denoising_No_filters(img, slide_dimensions, magnification, dictx)
        out = np.zeros_like(img)
        print("cleaning image at high mignification")
        mask = mask.astype(np.bool)
        out[mask] = img[mask]
        out = np.where(out != [0, 0, 0], out, [255, 255, 255])
        print("cleaning WSI done")
        #     cv2.imwrite("/home/pagenet2/PageNet2/Data Preprocessing Pipeline/WSI_Precessing_test/cleanedimages/%s/cleanedsmallf.png"%(inputsvs.split("/")[-1][:-4]),out)
        garbage_collector()
        print("exisiting cleaning")
    return out
