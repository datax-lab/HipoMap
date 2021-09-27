import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from numpy import interp
from sklearn import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Input, Flatten, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from hipomap.utils import relu, read_wsi, stain_remover_small_patch_remover


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


def extracting_patches(input_svs, magnification, patch_size, intermediate_layer_model=None):
    """
    For a given slide, a patch is extracted from a given condition(magnification, patch_size).
    GradCAM is applied to the value calculated through the last conv layer of the pre-trained model.
    The generated 2D array is converted into a 1D array and sorted in descending order.
    The calculated 1D arrays for each patches are combined(2D array) and then sorted in descending order.
    The value of the generated 2D array is called a HipoMap (graphical representation map).
    """
    all_patches = []

    if intermediate_layer_model is None:
        print("Error : Intermediate model not provided as parameter")
    else:
        slide, _ = read_wsi(input_svs, magnification)
        for i in range(int(len(slide[0]) / 299)):
            for j in range(int(len(slide) / 599)):
                center_point = ((j * 299 + (patch_size[0] / 2)), (i * 299 + (patch_size[1] / 2)))
                sample_img = slide[int(center_point[0] - patch_size[0] / 2):int(center_point[0] + patch_size[0] / 2),
                             int(center_point[1] - patch_size[1] / 2): int(center_point[1] + patch_size[1] / 2)]
                patches = stain_remover_small_patch_remover(sample_img, patch_size)
                if patches is not None:
                    patches = patches / 255
                    patches = np.expand_dims(patches, axis=0)
                    intermediate_output, y_pred = intermediate_layer_model.predict(patches)
                    _, width, height, _ = intermediate_output.shape
                    y_pred = y_pred[0][0]
                    alpha_patch = np.zeros((1, width, height))
                    for t in range(len(intermediate_output[0][0][0])):
                        grad = np.gradient(intermediate_output[:, :, :, t].flatten(), abs(1 - y_pred + 0.0001)) / 64
                        alpha = sum(grad)
                        alpha_patch += alpha * intermediate_output[:, :, :, t]
                    alpha_patch = relu(alpha_patch)
                    all_patches.append(sorted(alpha_patch.flatten(), reverse=True))
        all_patches = np.sort(np.array(all_patches), axis=0)[::-1]

    return all_patches


def generate_hipomap(input_path, output_path, magnification="20x", patch_size=(299, 299),
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
    input_path: str
        The path of directory where the whole slide image(.svs) files which wanted to create HipoMpa are located.
    output_path: str
        The path of directory to save the hipomap file(.npy).
    magnification: str [default="20x"]
        The magnification to be applied when creating hipomap.
        In more detail, to deal with the patch to be extracted and predicted.
    patch_size: tuple(int, int) [default=(299,299)]
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
        list_wsi = os.listdir(input_path)
        for wsi in list_wsi:
            if wsi[-3:] == 'svs':
                print("start " + wsi)
                data = input_path + wsi
                representation_map = extracting_patches(data, magnification, patch_size, intermediate_layer_model)
                print("done")
                save_path = output_path + wsi[:-4]
                np.save(save_path, representation_map)


def generate_classification_model(top, activation_map_size):
    """
    Build a model for classifying the generated hipomap into Cancer/Normal.

    Parameters
    ----------
    top: int
        The number of top patches which wanted to deal with K.
    activation_map_size: int
        The width of hipomap ndarray.
        The activation map size of the last conv layer from model which generated hipomap
        (activation map's width * height).
    Returns
    -------

    """
    input_ = Input(shape=(top, activation_map_size, 1))
    out = Conv2D(128, kernel_size=(2, 2), activation='relu')(input_)
    out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.3)(out)
    out = Conv2D(64, kernel_size=(2, 2), activation='relu')(out)
    out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.3)(out)
    out = Conv2D(32, kernel_size=(2, 2), activation='relu')(out)
    out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.3)(out)
    out = Conv2D(16, kernel_size=(2, 2), activation='relu')(out)
    out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.3)(out)
    out = Flatten()(out)
    out = Dense(units=1024, activation='relu')(out)
    out = Dense(units=1, activation='sigmoid')(out)

    return Model(input_, out)


class HipoClass:
    """
    HipoClass is to classify the Whole-Slide Image by using graphical representation map through hipoMap.

    The main contributions of hipomap are as follows:

    * Creating graphical representation maps as feature extraction of WSI.

    * Efficient training of the model without ROI annotations.

    * Allowing easier interpretation on findings of morphological patterns.

    Parameters
    ----------
    K: int [default=50]
        top K patches.
    activation_size: int [default=64]
        The Number of pixels in the activation map of the last con layer of the pre-trained model (width * height).
        The pre-trained model is a model that has already been trained on a patch basis and used to create a graphical
        presentation map.
    is_model: bool [default=False]
        If True, HipoClass uses the model provided by the user when classifying the representation map.
        If False, HipoClass builds a model for classifying the representation map.
    model_path: None of str [default=None]
        The path of pre-trained model for classifying representation map.
        If ismodel is False, the value of the modelpath should be set to None.
        If ismodel is True, the value of the modelpath must be set the str value where the model is located.

    Attributes
    ----------
    model: keras.Model
        Keras model for classify Cancer/Normal from representation map.


    Examples
    ----------
    >>> #Generate Hipomap (Representation map)
    >>> from hipomap.core import generate_hipomap
    >>> generate_hipomap(input_path="./Dataset/Slide/", output_path='./Dataset/Hipomap/', magnification="20x",
    >>>                  patch_size=(256, 256), model=None, layer_name=None)

    >>> #Classify Hipomaps to Cancer/Normal
    >>> from hipomap.core import HipoClass
    >>> hipo = HipoClass(K=50, activation_size=64, ismodel=False)
    >>> train, valid, test = hipo.split('./splitbase.csv', "/Dataset/Normal/", "/Dataset/Cancer/")
    >>> hipo.fit(train, valid, lr=0.01, epoch=50, batch_size=1)

    >>> #Prediction by TestDataset
    >>> testX, label = test[0], test[1]
    >>> prediction = hipo.predict_with_test(testX)
    >>> tpr, fpr, auc = hipo.evaluae_score(label, prediction)
    """

    def __init__(self, K=50, activation_size=64, is_model=False, model_path=None):
        self.K = K
        self.activation_size = activation_size
        self.is_model = is_model

        if is_model and model_path is not None:
            self.model = load_model(model_path)
        else:
            self.model = generate_classification_model(self.K, self.activation_size)

    def fit(self, train_set, valid_set, lr=0.01, epoch=50, batch_size=1):
        """
        Fit the classification model with Train, Validation dataset.

        Parameters
        ----------
        train_set: 2D array [rep_map, label]
            Train dataset with representation map, label.
            rep_map is array (n_samples, n_pixels)
            if label is 0, sample's class is normal, or sample's class is cancer.
        valid_set: 2D array [rep_map, label]
            Validation dataset with representation map, label.
        lr: float, default=0.01
            learning rate.
        epoch: int, default=50
        batch_size: int, default=1
            The number of batch.

        Attributes
        ----------
        mean: np.ndarray
            The array of mean values for each pixel value of the representation maps in train dataset.
        std: np.ndarray
            The standard deviation values for each pixel value of the representation maps in train dataset.


        Returns
        ----------
        keras.Model
        """

        train_X, train_y = train_set
        valid_X, valid_y = valid_set

        self.mean = np.mean(train_X, axis=0)
        self.std = np.std(train_X, axis=0)

        train_X = (train_X - self.mean) / self.std
        train_X = np.reshape(train_X, (len(train_X), self.K, self.activation_size))
        train_X = np.expand_dims(train_X, axis=3)
        train_y = np.array(train_y)

        valid_X = (valid_X - self.mean) / self.std
        valid_X = np.reshape(valid_X, (len(valid_X), self.K, self.activation_size))
        valid_X = np.expand_dims(valid_X, axis=3)
        valid_y = np.array(valid_y)

        SGD = optimizers.SGD(lr=lr)

        self.model.compile(optimizer=SGD, loss="binary_crossentropy", metrics=['accuracy', tf.keras.metrics.AUC()])
        self.model.fit(train_X, train_y, batch_size=batch_size, epochs=epoch, validation_data=(valid_X, valid_y))

        return self.model

    def predict_with_test(self, test_X):
        """
        Predict the class of WSI based Representation map with test Dataset from split method.

        Parameters
        ----------
        test_X: 2D array (n_samples, n_pixels)
            representation map in test set

        Returns
        ----------
        array
            prediction
        """
        test_X = (test_X - self.mean) / self.std
        test_X = np.reshape(test_X, (len(test_X), self.K, self.activation_size))
        test_X = np.expand_dims(test_X, axis=3)

        prediction = self.model.predict(test_X)

        return prediction

    def predict_with_sample(self, path):
        """
        Predict the class of WSI based Representation map.

        Parameters
        ----------
        path: str
            The path of directory where the representation map to be predicted is located.

        Returns
        ----------
        array
            prediction, classofHipo
        """
        list_hipo = os.listdir(path)
        prediction = []
        class_of_hipo = []
        for hipo in list_hipo:
            if hipo[-3:] == 'npy':
                data = path + hipo
                img = np.load(data)
                img = img[:self.K]
                if img.shape[0] >= self.K:
                    img = np.reshape(img, (1, self.K, self.activation_size))
                    img = np.expand_dims(img, axis=3)
                    pred = self.model.predict(img)
                    prediction.append(pred)
                    if pred > 0.5:
                        class_of_hipo.append("Cancer")
                    else:
                        class_of_hipo.append("Normal")

        return prediction, class_of_hipo

    def evaluate_score(self, label, prediction):
        """
        Evaluate score of the model with test dataset

        Parameters
        ----------
        label: array
                Real class(cancer or normal) of WSI.
        prediction: array
                     The prediction value of test set through model.


        Returns
        ----------
        float array, float
            tpr_score, mean_fpr, auc_score
        """
        label = np.array(label)
        fpr, tpr, threshold = metrics.roc_curve(label, prediction)
        mean_fpr = np.linspace(0, 1, 100)
        tpr_score = interp(mean_fpr, fpr, tpr)
        auc_score = metrics.auc(fpr, tpr)

        return tpr_score, mean_fpr, auc_score

    def split(self, split_csv, dir_normal, dir_cancer):
        """
        Split whole dataset to Train / Validation / Test.

        For splitting dataset, it requires csv file with content about which slides are divided into which datasets.

        Parameters
        ----------
        split_csv: str
            Path of split baseline file (.csv).
        dir_normal: str
            Path of Normal representation map dataset.
        dir_cancer: str
            Path of Cancer representation map dataset.

        Returns
        ----------
        tuple
            (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
        """
        splits = pd.read_csv(split_csv, header=None)
        trains = splits[splits[3] == 'train']
        valids = splits[splits[3] == 'valid']
        tests = splits[splits[3] == 'test']

        X_train, y_train = self._load_dataset(trains, dir_normal, dir_cancer)
        X_valid, y_valid = self._load_dataset(valids, dir_normal, dir_cancer)
        X_test, y_test = self._load_dataset(tests, dir_normal, dir_cancer)

        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    def _load_dataset(self, base, dir_normal, dir_cancer):
        """
        The method for loading stored representation maps with labels based on baseline.
        """
        list_X, list_y = [], []
        normal_set = os.listdir(dir_normal)
        cancer_set = os.listdir(dir_cancer)

        for data, label in zip(base[1], base[2]):
            if label == 'normal':
                for normal in normal_set:
                    if data in normal:
                        img = np.load(dir_normal + normal)
                        if img.shape[0] >= self.K:
                            list_X.append(img[:self.K].flatten())
                            list_y.append(0)
                        break
            else:
                for cancer in cancer_set:
                    if data in cancer:
                        img = np.load(dir_cancer + cancer)
                        if img.shape[0] >= self.K:
                            list_X.append(img[:self.K].flatten())
                            list_y.append(1)
                        break

        return np.array(list_X), np.array(list_y)
