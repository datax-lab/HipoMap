import os

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Model
from hipomap.utils import readWSI, stainremover_small_patch_remover1



def generateHipoMap(inputpath, outputpath, magnification="20x", patch_size=(299, 299),
                    model=None, layer_name=None):
    """
    Generate and Save graphical representation map (hipomap - Histopathology Map) by using last conv layer from pre-trained model.
    The pre-trained model is a model that has already been trained on a patch basis and used to create a graphical presentation map.

    hipomap is the framework for generating disease-specific graphical representation map from each Whole Slide Image.

    The main contributions of hipomap are as follows:

    * Creating graphical representation maps as feature extraction of WSI.

    * Efficient training of the model without ROI annotations.

    * Allowing easier interpretation on findings of morphological patterns.

    Parameters
    ----------
    inputpath: str
        The path of directory where the whole slide image(.svs) files which wanted to create HipoMpa are located.
    outputpath: str
        The path of directory to save the hipomap file(.npy).
    magnification: str [default="20x"]
        The magnification to be applied when creating hipomap.
        In more detail, to deal with the patch to be extracted and predicted.
    patch_size: tuple(width x height) [default=(299,299)]
        The size of the patch to be extracted from the Whole Slide image.
    model: keras.Model
        The model trained in advance on a patch basis.
    layer_name: str
        The name of the last convolutional layer in model

    """
    if model is None:
        print("Error : Model not provided as parameter")
    if layer_name is None:
        print("Error : layer_name not provided as parameter")
    else:
        intermediate_layer_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
        list_wsi = os.listdir(inputpath)
        for wsi in list_wsi:
            if wsi[-3:] == 'svs':
                print("start " + wsi)
                data = inputpath + wsi
                repmap = extractingPatches(data, magnification, patch_size, intermediate_layer_model)
                print("done")
                save_path = outputpath + wsi[:-4]
                np.save(save_path, repmap)


def extractingPatches(inputsvs, magnification, patch_size, intermediate_layer_model=None):
    """
    For a given slide, a patch is extracted from a given condition(magnificaiton, patch_size).
    GradCAM is applied to the value calculated through the last conv layer of the pre-trained model.
    The generated 2D array is converted into a 1D array and sorted in descending order.
    The calculated 1D arrays for each patches are combined(2D array) and then sorted in descending order.
    The value of the generated 2D array is called a HipmoMap (graphical representation map).
    """
    if intermediate_layer_model is None:
        print("Error : Intermediate model not provided as parameter")
    else:
        slide, _ = readWSI(inputsvs, magnification)
        ALL = []
        for i in range(int(len(slide[0]) / 299)):
            for j in range(int(len(slide) / 599)):
                centerpoint = ((j * 299 + (patch_size[0] / 2)), (i * 299 + (patch_size[1] / 2)))
                sample_img = slide[int(centerpoint[0] - patch_size[0] / 2):int(centerpoint[0] + patch_size[0] / 2),
                             int(centerpoint[1] - patch_size[1] / 2): int(centerpoint[1] + patch_size[1] / 2)]
                patchs = stainremover_small_patch_remover1(sample_img, patch_size)
                if patchs is None:
                    None
                else:
                    patchs = patchs / 255
                    patchs = np.expand_dims(patchs, axis=0)
                    intermediate_output, y_pred = intermediate_layer_model.predict(patchs)
                    _, width, height, _ = intermediate_output.shape
                    y_pred = y_pred[0][0]
                    alpha_patch = np.zeros((1, width, height))
                    for t in range(len(intermediate_output[0][0][0])):
                        grad = np.gradient(intermediate_output[:, :, :, t].flatten(), abs(1 - y_pred + 0.0001)) / 64
                        alpha = sum(grad)
                        alpha_patch += alpha * intermediate_output[:, :, :, t]
                    alpha_patch = ReLU(alpha_patch)
                    ALL.append(sorted(alpha_patch.flatten(), reverse=True))
        ALL = np.sort(np.array(ALL), axis=0)[::-1]

    return ALL


def draw_represent(path, K, max_value=1000, save=False):
    """
    To check by displaying the generated representation map.

    Parameters
    ----------
    path: str
        The path of the directory in which the representation map file(.npy) is located
    K: int
       The number of top K patches to be displayed (The height of np.ndarray)
    max_value: int [default=1000]
        The maximum threshold of pixel values to be displayed on the screen.
        The pixel values exceeding the value of the max are expressed as the same value as max.
    save: bool [default=False]
        If save is True, saving the representation map image displayed on the screen as a png file in the path
    """
    list_represent = os.listdir(path)
    for rep in list_represent:
        print(path + rep)
        try:
            plt.clf()
            heat = np.load(path + rep)
            plt.title("Heatmap for " + rep[:-3], fontsize=20)
            plt.xlabel("pixel of activation map", fontsize=5)
            plt.ylabel("patches", fontsize=5)
            ax = sns.heatmap(heat[:K], vmax=max_value)
            plt.show()
            if save:
                plt.savefig(path + rep[:-4] + '.png')
            plt.pause(0.1)
        except:
            pass


def ReLU(x):
    return x * (x > 0)
