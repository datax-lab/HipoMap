from hipomap.model_rep import model_rep
import numpy as np
from tensorflow.keras import optimizers
import tensorflow as tf
from numpy import interp
from sklearn import metrics
import pandas as pd
import os


class HipoClass:

    def __init__(self, K):
        self.K = K

    def fit(self, train_set, valid_set, lr, epoch, batchsize, activation_size):
        self.activation_size = activation_size

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
        model = model_rep(self.K, self.activation_size)

        model.compile(optimizer=SGD, loss="binary_crossentropy", metrics=['accuracy', tf.keras.metrics.AUC()])
        model.fit(train_X, train_y, batch_size=batchsize, epochs=epoch, validation_data=(valid_X, valid_y))

        self.model = model

        return model

    def predict(self, test_X):
        test_X = (test_X - self.mean) / self.std
        test_X = np.reshape(test_X, (len(test_X), self.K, self.activation_size))
        test_X = np.expand_dims(test_X, axis=3)

        prediction = self.model.predict(test_X)

        return prediction

    def evaluate_score(self, label, prediction):
        label = np.array(label)
        fpr, tpr, threshold = metrics.roc_curve(label, prediction)
        mean_fpr = np.linspace(0, 1, 100)
        tpr_score = interp(mean_fpr, fpr, tpr)
        auc_score = metrics.auc(fpr, tpr)

        return tpr_score, mean_fpr, auc_score

    def split(self, split_csv, dir_normal, dir_cancer):
        splits = pd.read_csv(split_csv, header=None)
        trains = splits[splits[3] == 'train']
        valids = splits[splits[3] == 'valid']
        tests = splits[splits[3] == 'test']

        X_train, y_train = self.load_Dataset(trains, dir_normal, dir_cancer)
        X_valid, y_valid = self.load_Dataset(valids, dir_normal, dir_cancer)
        X_test, y_test = self.load_Dataset(tests, dir_normal, dir_cancer)

        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    def load_Dataset(self, base, dir_normal, dir_cancer):
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





