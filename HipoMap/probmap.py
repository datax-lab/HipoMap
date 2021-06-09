import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys

sys.path.append("..")
import seaborn as sns
import numpy as np
from WSI_Preprocessing.Preprocessing.WSI_Scanning import readWSI
from openslide import OpenSlide
import cv2
import gc


def garbage_collector():
    for j in range(2):
        n = gc.collect()
        return


def plot_generate(y_pred):
    plt = sns.heatmap(np.array(np.log([[y_pred]])), yticklabels=False, xticklabels=False, cmap='jet', vmin=np.log(0.2),
                      vmax=np.log(1), cbar=False).get_figure()
    plt.savefig("example.png")
    img = cv2.imread("example.png")
    img = np.where(img != [255, 255, 255], img, img[144, 144])
    img = img[10:20, 10:20]
    img = cv2.resize(img, (60, 60))
    garbage_collector()
    os.remove("example.png")
    return img


def extracting(inputsvs, Annotation=None,
               Annotatedlevel=0, Requiredlevel=0, heat=None):
    slide1, slidedim = readWSI(inputsvs, "20x", Annotation, Annotatedlevel, Requiredlevel)
    reconstrcutedimage = np.zeros_like(slide1)
    reconstrcutedimage = np.array(reconstrcutedimage, dtype='uint8')
    slide1 = np.array(slide1, dtype='uint8')
    reconstrcutedimagen = cv2.resize(reconstrcutedimage,
                                     (int(len(reconstrcutedimage[0]) / 5), int(len(reconstrcutedimage) / 5)),
                                     interpolation=cv2.INTER_AREA)
    slide1n = cv2.resize(slide1, (int(len(slide1[0]) / 5), int(len(slide1) / 5)), interpolation=cv2.INTER_AREA)
    heat1 = sorted(np.array(heat), key=lambda x: x[0], reverse=True)

    for b in range(len(heat1)):
        j = heat1[b][1] / 5
        i = heat1[b][2] / 5
        img = plot_generate(heat1[b][0])
        if b < 100:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "%s" % (b + 1)
            cv2.putText(img, text, (0 + 5, len(img) - 15), font, 1, (10, 10, 10), 2)
        centerpoint20x = ((j * 299), (i * 299))
        try:
            reconstrcutedimagen[int(centerpoint20x[0] - 44 / 2):int(centerpoint20x[0] - 44 / 2 + 60),
            int(centerpoint20x[1] - 14 / 2): int(centerpoint20x[1] - 14 / 2 + 60)] = img
        except:
            None

    reconstrcutedimagen[np.where((reconstrcutedimagen == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    return reconstrcutedimagen, slide1n


def generating_probmap(path_data, path_prob, path_save):
    list_slide = os.listdir(path_data)
    list_heat = os.listdir(path_heat)

    for data in list_slide:
        print(data)
        slide = path_data + data

        for heat_ in list_heat:
            print(heat_)
            if heat_[:-8] == data[:-4]:
                slide_ = OpenSlide(slide)
                slide_dimensions = slide_.level_dimensions
                if len(slide_dimensions) == 3:
                    heat = np.load(path_heat + heat_)
                    reconstructedimage, slide1 = extracting(slide, Annotation=None, Annotatedlevel=0, Requiredlevel=0,
                                                            heat=heat)
                    cv2.imwrite(path_save + heat_[-12:-8] + ".png", reconstructedimage)
                    cv2.imwrite(path_save + heat_[-12:-8] + "_org.png", slide1)
                else:
                    heat = np.load(path_heat + heat_)
                    reconstrcutedimage, slide1 = extracting(slide, Annotation=None, Annotatedlevel=0, Requiredlevel=1,
                                                            heat=heat)
                    cv2.imwrite(path_save + heat_[-12:-8] + ".png", reconstructedimage)
                    cv2.imwrite(path_save + heat_[-12:-8] + "_org.png", slide1)