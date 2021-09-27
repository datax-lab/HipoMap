import os

import numpy as np
from PIL import ImageFile
from openslide import OpenSlide
from tensorflow.keras.models import load_model

from hipomap.utils import read_wsi, stain_remover_small_patch_remover

ImageFile.LOAD_TRUNCATED_IMAGES = True


def scoring_probmap(path_model, path_data, path_save, patch_size=(299, 299)):
    """
    Generate and Save an array containing the probability of each patch predicted from the model pre-trained in patch and the location information of patch.

    Parameters
    ----------
    path_model: str
        The path where the model(.pth) trained by patches is located.
    path_data: str
        The path of directory where the Whole Slide Image is located.
    path_save: str
        The path for saving the result file(.npy).
    patch_size: tuple [default=(299,299)]
        The size of the patch to be extracted from the Whole Slide image.
    """

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
            probmap = extracting(data, patch_size=patch_size, model=model)
        else:
            probmap = extracting(data, patch_size=patch_size, model=model)
        probmap_path = path_save + slide_[:-4] + "_new"
        np.save(probmap_path, probmap)


def extracting(input_svs, patch_size=(299, 299), model=None):
    """
    Patches are extracted from the whole slide to predict the probability for each patch.
    The predicted probability and the location of the extracted patch on the slide are saved in a tuple
    (probability, X of slide, Y of slide).
    """
    slide1, slidedim = read_wsi(input_svs, "20x")
    ALL_P1 = []
    for i in range(int(len(slide1[0]) / 299)):
        for j in range(int(len(slide1) / 299)):

            centerpoint20x = ((j * 299 + (patch_size[0] / 2)), (i * 299 + (patch_size[1] / 2)))
            sample_img = slide1[int(centerpoint20x[0] - patch_size[0] / 2):int(centerpoint20x[0] + patch_size[0] / 2),
                         int(centerpoint20x[1] - patch_size[1] / 2): int(centerpoint20x[1] + patch_size[1] / 2)]
            patchs = stain_remover_small_patch_remover(sample_img, patch_size)
            if patchs is None:
                None
            else:
                patchs = patchs / 255
                patchs = np.expand_dims(patchs, axis=0)
                y_pred = model.predict(patchs)
                ALL_P1.append((y_pred[0][0], j, i))
    return ALL_P1
