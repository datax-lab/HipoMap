from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from tensorflow.keras.layers import  BatchNormalization,Activation,Dropout,SpatialDropout2D
from tensorflow.keras.regularizers import l2

class IndiviualCNN():

    def __init__(self, filters, num_row, num_col, name, strides=(1, 1)):
        self.strides = strides
        self.filters = filters
        self.num_row = num_row
        self.num_col = num_col
        self.name = name
        self.conv_name = self.name + '_conv'
        self.bn_name = self.name + '_bn'
        self.Padding = "same"
        self.dilation_rate = (1, 1)
        self.eps = 0.001
        self.conv_name = self.name + '_conv'
        self.Kernel_regularizer = l2(0.02)
    def bulid(self, inputlayer):
        Dilatedlayer = Conv2D(self.filters, (self.num_row, self.num_col), dilation_rate=self.dilation_rate,
                              strides=self.strides, padding=self.Padding, use_bias=False, name=self.conv_name)(
            inputlayer)
        BatchNor = BatchNormalization(axis=1, scale=True, epsilon=self.eps, name=self.bn_name)(Dilatedlayer)

        BatchNor = SpatialDropout2D(0.3)(BatchNor)
        Act = Activation('relu', name=self.name)(BatchNor)
        return Act
class Indiviualdilation():

    def __init__(self, filters,num_row, num_col,name):
        self.filters = filters
        self.num_row = num_row
        self.num_col = num_col
        self.name = name
        self.conv_name = self.name+'_conv'
        self.bn_name = self.name+'_bn'
        self.Padding = "valid"
        self.dilation_rate = (2,2)
        self.eps = 0.001
        self.conv_name = self.name+'_conv'
        self.Kernel_regularizer = l2(0.02)
        
    def bulid(self,inputlayer):
    
        Dilatedlayer = Conv2D(self.filters, (self.num_row,self.num_col),dilation_rate = self.dilation_rate,strides=(1,1),padding=self.Padding,use_bias=False,name = self.conv_name, kernel_regularizer = self.Kernel_regularizer)(inputlayer)
        BatchNor = BatchNormalization(axis = 1, scale= False,  name = self.bn_name)(Dilatedlayer)
        BatchNor = SpatialDropout2D(0.4)(BatchNor)
        Act = Activation('relu', name= self.name)(BatchNor)
        return Act
        
class Dilationlayer():
    
    def __init__(self, name):
        
        self.name = name
        
    def bulid(self,inputmodel):
        Dil1 = Indiviualdilation(32,3,3,name = self.name+"dil1").bulid(inputmodel)
        Dil2 = Indiviualdilation(32,3,3,name = self.name+"dil2").bulid(Dil1)
        Dil3 = Indiviualdilation(64,3,3,name = self.name+"dil3").bulid(Dil2)
        Max1 = MaxPooling2D((3, 3),strides=(2, 2),padding='same')(Dil3)
        Dil4 = Indiviualdilation(64,1,1,name = self.name+"dil4").bulid(Max1)
        Dil5 = Indiviualdilation(80,3,3,name = self.name+"dil5").bulid(Dil4)
        Max2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Dil5)
        return Max2


class CNNlayer():

    def __init__(self, name):
        self.name = name


    def bulid(self, inputmodel):
        Dil1 = IndiviualCNN(32, 3, 3, name=self.name + "dil1").bulid(inputmodel)
        Dil2 = IndiviualCNN(32, 3, 3, name=self.name + "dil2").bulid(Dil1)
        Dil3 = IndiviualCNN(64, 3, 3, name=self.name + "dil3").bulid(Dil2)
        Max1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Dil3)
        Dil4 = IndiviualCNN(64, 1, 1, name=self.name + "dil4").bulid(Max1)
        Dil5 = IndiviualCNN(80, 3, 3, name=self.name + "dil5").bulid(Dil4)
        Max2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Dil5)
        return Max2
