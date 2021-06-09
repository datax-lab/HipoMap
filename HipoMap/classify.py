from HipoMap.model_rep import model_rep
import numpy as np
from tensorflow.keras import optimizers
import tensorflow as tf
from scipy import interp
from sklearn import metrics
import pandas as pd
import os


class HipoClass:
    """
    HipoClass is for predicting Class of representation map (Whole-Slide Image based).

    Parameters
    ----------
    K: int
        top K patches.


    Attributes
    ----------
    activation_size: int
        Number of pixels in last conv layer's activation map (width * height).



    Examples
    ----------
    >>> from HipoClassify imort HipoClass
    >>> hipo = HipoClass(K=50)
    >>> train, valid, test = hipo.split('./splitbase.csv', "/Dataset/Normal/", "/Dataset/Cancer/")
    >>> hipo.fit(train, valid, lr=0.01, epoch=50, batchsize=1, activation_size=64)

    >>> testX, label = test[0], test[1]
    >>> prediction = hipo.predict(testX)
    >>> tpr, fpr, auc = hipo.evaluae_score(label, prediction)
    """

    def __init__(self, K, activation_size, ismodel, modelpath):
        self.K = K
        self.activation_size = activation_size
        self.ismodel = ismodel
        if ismodel:
            self.model = load_model(modelpath)
        else:
            self.model = model_rep(self.K, self.activation_size)

    def fit(self, train_set, valid_set, lr, epoch, batchsize):
        """
        Fit the classification model with Train, Validation dataset.

        Parameters
        ----------
        train_set: array
                    Train dataset with feature, label.
        valid_set: array
                    Validation dataset with feature, label.
        lr: float
             learning rate.
        epoch: int
        batchsize: int
                     The number of batch.
        activation_size: int
                          The number of pixels in last conv layer's activation map (width * height).

        Attributes
        ----------
        mean:
            The average value of train dataset.
        std: array
              The standard deviation of train dataset.
        model:
            The classification model learned with representatino map.


        Returns
        ----------
        model: model
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
        Predict the class of WSI based Representation map.

        Parameters
        ----------
        train_X: array
                  Test feature dataset.

        Returns
        ----------
        prediction: float array
        """
        test_X = (test_X - self.mean) / self.std
        test_X = np.reshape(test_X, (len(test_X), self.K, self.activation_size))
        test_X = np.expand_dims(test_X, axis=3)

        prediction = self.model.predict(test_X)

        return prediction

    def predict_with_sample(self, path):
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
        tpr_score: float array
                    The value of tpr.
        mean_fpr: float array
                    The mean value of fpr.
        auc_score: float
                    The value of AUC(Area under curve).
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
        split_csv:
            Path of split baseline file (.csv).
        dir_normal:
            Path of Normal representation map dataset.
        dir_cancer:
            Path of Cancer representation map dataset.

        Returns
        ----------
        (X_train, y_train):
            A tuple of feature, label set about trainset.
        (X_valid, y_valid):
            A tuple of feature, label set about validation set.
        (X_test, y_test):
            A tuple of feature, label set about testset.
        """
        splits = pd.read_csv(split_csv, header=None)
        trains = splits[splits[3] == 'train']
        valids = splits[splits[3] == 'valid']
        tests = splits[splits[3] == 'test']

        X_train, y_train = self.load_Dataset(trains, dir_normal, dir_cancer)
        X_valid, y_valid = self.load_Dataset(valids, dir_normal, dir_cancer)
        X_test, y_test = self.load_Dataset(tests, dir_normal, dir_cancer)

        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    def load_Dataset(self, base, dir_normal, dir_cancer):
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