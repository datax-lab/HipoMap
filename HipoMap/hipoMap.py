import numpy as np
import os
from WSI_Preprocessing.Preprocessing import WSI_Scanning, Utilities
from tensorflow.keras.models import Model, load_model
import seaborn as sns
import matplotlib.pylab as plt


def ReLU(x):
    return x * (x > 0)


def extractingPatches(inputsvs, magnification, patch_size, Annotation=None,
                      Annotatedlevel=0, Requiredlevel=0, intermediate_layer_model=None):
    slide, _ = WSI_Scanning.readWSI(inputsvs, magnification, Annotation, Annotatedlevel, Requiredlevel)
    ALL = []
    for i in range(int(len(slide[0]) / 299)):
        for j in range(int(len(slide) / 599)):
            centerpoint = ((j * 299 + (patch_size[0] / 2)), (i * 299 + (patch_size[1] / 2)))
            sample_img = slide[int(centerpoint[0] - patch_size[0] / 2):int(centerpoint[0] + patch_size[0] / 2),
                         int(centerpoint[1] - patch_size[1] / 2): int(centerpoint[1] + patch_size[1] / 2)]
            patchs = Utilities.stainremover_small_patch_remover1(sample_img, patch_size)
            if patchs is None:
                None
            else:
                patchs = patchs / 255
                patchs = np.expand_dims(patchs, axis=0)
                intermediate_output, y_pred = intermediate_layer_model.predict(patchs)
                _, width, height, _ = intermediate_output.shape
                y = y_pred[0][0]
                al_patch = np.zeros((1, width, height))
                for t in range(len(intermediate_output[0][0][0])):
                    grad = np.gradient(intermediate_output[:, :, :, t].flatten(), abs(1 - y + 0.0001)) / 64
                    al = sum(grad)
                    al_patch += al * intermediate_output[:, :, :, t]
                al_patch = ReLU(al_patch)
                ALL.append(sorted(al_patch.flatten(), reverse=True))
    ALL = np.sort(np.array(ALL), axis=0)[::-1]
    return ALL


def generateHipoMap(inputpath, outputpath, magnification="20x", patch_size=(299, 299),
                    model=None, layer_name=None, Annotation=None, Annotatedlevel=0, Requiredlevel=0):
    if model == None:
        print("Error : Model not provided as parameter")
    if layer_name == None:
        print("Error : layer_name not provided as parameter")
    else:
        intermediate_layer_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
        list_wsi = os.listdir(inputpath)
        for wsi in list_wsi:
            if wsi[-3:] == 'svs':
                print("start " + wsi)
                data = inputpath + wsi
                repmap = extractingPatches(data, magnification, patch_size, Annotation, Annotatedlevel, Requiredlevel,
                                           intermediate_layer_model)
                print("done")
                save_path = outputpath + wsi[:-4]
                np.save(save_path, repmap)


def draw_represent(path, K, max_value=1000, save=False):
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
