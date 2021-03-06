import gc
import os

import cv2
import numpy as np
import seaborn as sns
from PIL import ImageFile
from openslide import OpenSlide

from hipomap.utils import read_wsi

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def extracting(input_svs, heat=None):
    slide1, slidedim = read_wsi(input_svs, "20x")
    reconstructedimage = np.zeros_like(slide1)
    reconstructedimage = np.array(reconstructedimage, dtype='uint8')
    slide1 = np.array(slide1, dtype='uint8')
    reconstructedimagen = cv2.resize(reconstructedimage,
                                     (int(len(reconstructedimage[0]) / 5), int(len(reconstructedimage) / 5)),
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
            reconstructedimagen[int(centerpoint20x[0] - 44 / 2):int(centerpoint20x[0] - 44 / 2 + 60),
            int(centerpoint20x[1] - 14 / 2): int(centerpoint20x[1] - 14 / 2 + 60)] = img
        except:
            None

    reconstructedimagen[np.where((reconstructedimagen == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    return reconstructedimagen, slide1n


def generating_probmap(path_data, path_prob, path_save):
    """
    Visualize by displaying the ranking along with the color (heatmap) on the whole slide image according to the ranking of probability by patches.
    save origin slide image and heatmap with probability ranking image to png file.

    Parameters
    ----------
    path_data: str
        The path of directory where the whole slide image is located.
    path_prob: str
        The path of directory where probability score array is loacated.
        The probability score array is generated by scoring.scoring_probmap()
    path_save: str
        The path for save origin image and heatmap with prob-ranking image.
    """
    list_slide = os.listdir(path_data)
    list_heat = os.listdir(path_prob)

    for data in list_slide:
        print(data)
        slide = path_data + data

        for heat_ in list_heat:
            print(heat_)
            if heat_[:-8] == data[:-4]:
                slide_ = OpenSlide(slide)
                slide_dimensions = slide_.level_dimensions
                if len(slide_dimensions) == 3:
                    heat = np.load(path_prob + heat_)
                    reconstructedimage, slide1 = extracting(slide, heat=heat)
                    cv2.imwrite(path_save + heat_[:-8] + ".png", reconstructedimage)
                    cv2.imwrite(path_save + heat_[:-8] + "_org.png", slide1)
                else:
                    heat = np.load(path_prob + heat_)
                    reconstructedimage, slide1 = extracting(slide, heat=heat)
                    cv2.imwrite(path_save + heat_[:-8] + ".png", reconstructedimage)
                    cv2.imwrite(path_save + heat_[:-8] + "_org.png", slide1)
