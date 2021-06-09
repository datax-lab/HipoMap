import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.models import load_model

import sys
sys.path.append("")
import seaborn as sns
import numpy as np
from WSI_Preprocessing.Preprocessing.WSI_Scanning import readWSI
from WSI_Preprocessing.Preprocessing.Utilities import stainremover_small_patch_remover1
from openslide import OpenSlide
import cv2


def plot_score(y_pred):
    plt = sns.heatmap(np.array([[y_pred]]), yticklabels=False,
                      xticklabels=False, cmap='coolwarm',
                      vmin=0, vmax=1, cbar=False).get_figure()
    plt.savefig("example.png")
    img = cv2.imread("example.png")
    img = np.where(img != [255, 255, 255], img, img[144, 144])
    img = cv2.resize(img, (299, 299))
    os.remove("example.png")
    return img


def extracting(inputsvs, patch_size=(299, 299), Annotation=None, Annotatedlevel=0, Requiredlevel=0, model=None):
    slide1, slidedim = readWSI(inputsvs, "20x", Annotation, Annotatedlevel, Requiredlevel)
    ALL_P1 = []
    for i in range(int(len(slide1[0]) / 299)):
        for j in range(int(len(slide1) / 299)):

            centerpoint20x = ((j * 299 + (patch_size[0] / 2)), (i * 299 + (patch_size[1] / 2)))
            sample_img = slide1[int(centerpoint20x[0] - patch_size[0] / 2):int(centerpoint20x[0] + patch_size[0] / 2),
                         int(centerpoint20x[1] - patch_size[1] / 2): int(centerpoint20x[1] + patch_size[1] / 2)]
            patchs = stainremover_small_patch_remover1(sample_img, patch_size)
            if patchs is None:
                None
            else:
                patchs1 = patchs / 255
                patchs1 = np.expand_dims(patchs1, axis=0)
                y_pred = model.predict(patchs1)
                ALL_P1.append((y_pred[0][0], j, i))
    return ALL_P1


def scoring_probmap(path_model, path_data, path_save, patch_size=(299, 299)):
    model = load_model(path_model)
    list_data = os.listdir(path_data)
    list_slide = []
    for data in list_data:
        if data[-3:] == 'svs':
            list_slide.append(data)

    for slide_ in list_slide:
        print(slide_)
        data = path_data + slide_
        slide = OpenSlide(data)
        slide_dimensions = slide.level_dimensions
        if len(slide_dimensions) == 3:
            probmap = extracting(data, patch_size=patch_size,
                                 Annotation=None, Annotatedlevel=0, Requiredlevel=0, model=model)
        else:
            probmap = extracting(data, patch_size=patch_size,
                                 Annotation=None, Annotatedlevel=0, Requiredlevel=1, model=model)
        probmap_path = path_save + slide_[:-4] + "_new"
        np.save(probmap_path, probmap)
