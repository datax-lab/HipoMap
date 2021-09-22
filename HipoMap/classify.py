from HipoMap.model_rep import model_rep
import numpy as np
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.models import load_model
from numpy import interp
from sklearn import metrics
import pandas as pd
import os


class HipoClass:
    """
    HipoClass is to classify the Whole-Slide Image by using graphical representation map through hipoMap.

    The main contributions of HipoMap are as follows:

    * Creating graphical representation maps as feature extraction of WSI.

    * Efficient training of the model without ROI annotations.

    * Allowing easier interpretation on findings of morphological patterns.

    Parameters
    ----------
    K: int [default=50]
        top K patches.
    activation_size: int [default=64]
        The Number of pixels in the activation map of the last con layer of the pre-trained model (width * height).
        The pre-trained model is a model that has already been trained on a patch basis and used to create a graphical presentation map.
    ismodel: bool [default=False]
        If True, HipoClass uses the model provided by the user when classifying the representation map.
        If False, HipoClass builds a model for classifying the representation map.
    modelpath: None of str [default=None]
        The path of pre-trained model for classifying representation map.
        If ismodel is False, the value of the modelpath should be set to None.
        If ismodel is True, the value of the modelpath must be set the str value where the model is located.

    Attributes
    ----------
    model: keras.Model
        Keras model for classify Cancer/Normal from representation map.



    Examples
    ----------
    >>> from HipoMap.classify import HipoClass
    >>> hipo = HipoClass(K=50, activation_size=64, ismodel=False)
    >>> train, valid, test = hipo.split('./splitbase.csv', "/Dataset/Normal/", "/Dataset/Cancer/")
    >>> hipo.fit(train, valid, lr=0.01, epoch=50, batchsize=1)

    >>> testX, label = test[0], test[1]
    >>> prediction = hipo.predict_with_test(testX)
    >>> tpr, fpr, auc = hipo.evaluae_score(label, prediction)
    """

    def __init__(self, K=50, activation_size=64, ismodel=False, modelpath=None):
        self.K = K
        self.activation_size = activation_size
        self.ismodel = ismodel
        if ismodel:
            self.model = load_model(modelpath)
        else:
            self.model = model_rep(self.K, self.activation_size)

    def fit(self, train_set, valid_set, lr=0.01, epoch=50, batchsize=1):
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
        batchsize: int, default=1
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
        self.model.fit(train_X, train_y, batch_size=batchsize, epochs=epoch, validation_data=(valid_X, valid_y))

        return self.model

    def predict_with_test(self, test_X):
        """
        Predict the class of WSI based Representation map with test Dataset from split method.

        Parameters
        ----------
        test_X: 2D array (n_samples, n_pixels)
            representation map in testset

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
        classofHipo = []
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
                        classofHipo.append("Cancer")
                    else:
                        classofHipo.append("Normal")

        return prediction, classofHipo

    def evaluate_score(self, label, prediction):
        """
        Evaludate score of the model with test dataset

        Parameters
        ----------
        label: array
                Real class(cancer or normal) of WSI.
        prediction: array
                     The prediction value of testset through model.


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