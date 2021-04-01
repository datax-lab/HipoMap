import numpy as np
import tensorflow as tf
from Dilationlayer import Dilationlayer,CNNlayer
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,GlobalMaxPooling2D,SpatialDropout2D
import os
from InceptionA import InceptionA
from InceptionB import InceptionB
from InceptionC import InceptionC
from InceptionD import InceptionD
from InceptionE import InceptionE
from tensorflow.keras.layers import  BatchNormalization,Activation,concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
        
class CATNet():

    def __init__(self, name):
        self.name = name

        
        
    def bulid(self,input_img):
        model  = Dilationlayer(self.name+"baselayer").bulid(input_img)
        # model = Dilationlayer(self.name + "baselayer1").bulid(input_img)
        # model = Dropout(0.2)(model)
        # model = Dilationlayer(self.name + "baselayer1").bulid(model)
        # with tf.device('/device:GPU:1'):
        # model = Dropout(0.25)(model)
        model  = InceptionA(self.name+"mixedA1").bulid(model)
        # # model = Dropout(0.2)(model)
        # # # model = Dropout(0.25)(model)
        # # model  = InceptionA(self.name+"mixedA2").bulid(model)
        # # # model  = InceptionA(self.name+"mixedA3").bulid(model)
        model  = InceptionB(self.name+"mixedB1").bulid(model)
        # # model = Dropout(0.2)(model)
        # # # model = Dropout(0.25)(model)
        # # # with tf.device('/device:GPU:2'):
        model  = InceptionC(self.name+"mixedC1",128).bulid(model)
        model  = InceptionC(self.name+"mixedC2",160).bulid(model)
        # model = Dropout(0.25)(model)
        model  = InceptionC(self.name+"mixedC3",160).bulid(model)
        model  = InceptionC(self.name+"mixedC4",192).bulid(model)
        # # with tf.device('/device:GPU:3'):
        model  = InceptionD(self.name+"mixedD1").bulid(model)
        # # # model = Dropout(0.25)(model)
        model  = InceptionE(self.name+"mixedE1").bulid(model)
        model = Dropout(0.2)(model)
        # model  = InceptionE(self.name+"mixedE2").bulid(model)

        # with tf.device('/device:GPU:'):
        # model = Dropout(0.4)(model)
        model = Model(input_img,model)

        return model


class GoogleNet():

    def __init__(self, name):
        self.name = name

    def bulid(self, input_img):
        model = CNNlayer(self.name + "baselayer").bulid(input_img)
        # model = Dilationlayer(self.name + "baselayer1").bulid(input_img)
        # model = Dropout(0.2)(model)
        # model = Dilationlayer(self.name + "baselayer1").bulid(model)
        # with tf.device('/device:GPU:1'):
        # model = Dropout(0.25)(model)
        model  = InceptionA(self.name+"mixedA1").bulid(model)
        # # model = Dropout(0.2)(model)
        # # # model = Dropout(0.25)(model)
        model  = InceptionA(self.name+"mixedA2").bulid(model)
        model  = InceptionA(self.name+"mixedA3").bulid(model)
        model  = InceptionB(self.name+"mixedB1").bulid(model)
        # # model = Dropout(0.2)(model)
        # # # model = Dropout(0.25)(model)
        # # # with tf.device('/device:GPU:2'):
        model  = InceptionC(self.name+"mixedC1",128).bulid(model)
        model  = InceptionC(self.name+"mixedC2",160).bulid(model)
        # # model = Dropout(0.25)(model)
        model  = InceptionC(self.name+"mixedC3",160).bulid(model)
        model  = InceptionC(self.name+"mixedC4",192).bulid(model)
        # # with tf.device('/device:GPU:3'):
        model  = InceptionD(self.name+"mixedD1").bulid(model)
        # # # model = Dropout(0.25)(model)
        model  = InceptionE(self.name+"mixedE1").bulid(model)
        model = SpatialDropout2D(0.2)(model)
        model  = InceptionE(self.name+"mixedE2").bulid(model)

        # with tf.device('/device:GPU:'):
        # model = Dropout(0.4)(model)
        model = Model(input_img, model)

        return model
        

class CATNet2():
    
    def __init__(self, name):
        
        self.name = name

        

    def bulid(InputA,InputB):
        # with tf.device('/device:GPU:0'):
        CATNet_Track1 = CATNet("20xTrack").bulid(InputA)
        CATNet_Track1output = CATNet_Track1.output

        CATNet_Track2 = CATNet("5xTrack").bulid(InputB)

        CATNet_Track2output = CATNet_Track2.output
        # CATNet_Track2 = Dropout(0.4)(CATNet_Track2)
        CATNet_Track1flat = GlobalAveragePooling2D()(CATNet_Track1output)
        # CATNet_Track1flat = Flatten()(CATNet_Track1output)
        # CATNet_Track2flat = Flatten()(CATNet_Track2output)

        CATNet_Track2flat =  GlobalAveragePooling2D()(CATNet_Track2output)
        # CATNet_Track2flat = Flatten()(CATNet_Track2flat)
        combined = concatenate([CATNet_Track1flat, CATNet_Track2flat])
        # combinedflat = GlobalAveragePooling1D()(combined)
        # combined = Flatten()(combined)
        z = Dense(128, activation='tanh',kernel_regularizer = l2(0.04))(combined)
        # z = Dense(512, activation='tanh')(z)
        z = Dropout(0.2)(z)
        predictions = Dense(1, activation='sigmoid',kernel_regularizer = l2(0.04))(z)
        model = Model(inputs=[CATNet_Track1.input,CATNet_Track2.input], outputs=predictions)
        return model

    def bulid1(InputA):
        # with tf.device('/device:GPU:0'):
        CATNet_Track1 = CATNet("20xTrackCAT_NEt").bulid(InputA)
        CATNet_Track1output = CATNet_Track1.output

        # CATNet_Track2 = CATNet("5xTrack").bulid(InputB)
        #
        # CATNet_Track2output = CATNet_Track2.output
        # # CATNet_Track2 = Dropout(0.4)(CATNet_Track2)
        CATNet_Track1flat = GlobalAveragePooling2D()(CATNet_Track1output)
        # CATNet_Track1flat = Flatten()(CATNet_Track1output)
        # CATNet_Track2flat = Flatten()(CATNet_Track2output)

        # CATNet_Track2flat = GlobalAveragePooling2D()(CATNet_Track2output)
        # CATNet_Track2flat = Flatten()(CATNet_Track2flat)
        # combined = concatenate([CATNet_Track1flat])
        # combinedflat = GlobalAveragePooling1D()(combined)
        # combined = Flatten()(combined)
        z = Dense(256, activation='tanh', kernel_regularizer=l2(0.04))(CATNet_Track1flat)
        # z = Dense(512, activation='tanh')(z)
        z = Dropout(0.2)(z)
        predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(z)
        model = Model(inputs=[CATNet_Track1.input], outputs=predictions)
        return model
    def bulid2(InputA):
        # with tf.device('/device:GPU:0'):
        CATNet_Track1 = GoogleNet("20xTrack").bulid(InputA)
        CATNet_Track1output = CATNet_Track1.output

        # CATNet_Track2 = CATNet("5xTrack").bulid(InputB)
        #
        # CATNet_Track2output = CATNet_Track2.output
        # # CATNet_Track2 = Dropout(0.4)(CATNet_Track2)
        CATNet_Track1flat = GlobalAveragePooling2D()(CATNet_Track1output)
        # CATNet_Track1flat = Flatten()(CATNet_Track1output)
        # CATNet_Track2flat = Flatten()(CATNet_Track2output)

        # CATNet_Track2flat = GlobalAveragePooling2D()(CATNet_Track2output)
        # CATNet_Track2flat = Flatten()(CATNet_Track2flat)
        # combined = concatenate([CATNet_Track1flat])
        # combinedflat = GlobalAveragePooling1D()(combined)
        # combined = Flatten()(combined)
        z = Dense(256, activation='tanh', kernel_regularizer=l2(0.4))(CATNet_Track1flat)
        # z = Dense(512, activation='tanh')(z)
        z = Dropout(0.2)(z)
        predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.4))(z)
        model = Model(inputs=[CATNet_Track1.input], outputs=predictions)
        return model

# if __name__ = "__main__":
#     InputA = Input(shape=(299, 299,3))
#     InputB = Input(shape=(299, 299,3))
#     model = CATNet2.bulid(InputA, InputB)
        
