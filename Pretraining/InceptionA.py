from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D,AveragePooling2D,concatenate
from tensorflow.keras.layers import  BatchNormalization,Activation,Dropout,SpatialDropout2D
from tensorflow.keras.regularizers import l2

class IndiviualCNN():

    def __init__(self, filters,num_row, num_col,name):
        self.filters = filters
        self.num_row = num_row
        self.num_col = num_col
        self.name = name
        self.conv_name = self.name+'_conv'
        self.bn_name = self.name+'_bn'
        self.Padding = "same"
        self.dilation_rate = (1,1)
        self.eps = 0.001
        self.conv_name = self.name+'_conv'
        self.Kernel_regularizer = l2(0.02)



    def bulid(self,inputlayer):

        Dilatedlayer = Conv2D(self.filters, (self.num_row,self.num_col),dilation_rate = self.dilation_rate,strides=(1,1),padding=self.Padding,use_bias=False,name = self.conv_name, kernel_regularizer = self.Kernel_regularizer)(inputlayer)
        BatchNor = BatchNormalization(axis = 1, scale= False, name = self.bn_name)(Dilatedlayer)
        # BatchNor = Dropout(0.2)(BatchNor)
        BatchNor = SpatialDropout2D(0.4)(BatchNor)
        Act = Activation('tanh', name= self.name)(BatchNor)
        return Act
        

class InceptionA():
    
    def __init__(self, name):
        
        self.name = name

        

    def bulid(self,inputmodel):
        #branch1
        branch1x1 = IndiviualCNN(64,1,1,name = self.name+"inceptiona1x1").bulid(inputmodel)

        #branch2
        branch5x5_1 = IndiviualCNN(48,1,1,name = self.name+"inceptiona5x5_1").bulid(inputmodel)
        branch5x5_2 = IndiviualCNN(64,5,5,name = self.name+"inceptiona5x5_2").bulid(branch5x5_1)

        #branch3
        branch3x3_1 = IndiviualCNN(64,1,1,name = self.name+"inceptiona3x3_1").bulid(inputmodel)
        branch3x3_2 = IndiviualCNN(96,3,3,name = self.name+"inceptiona3x3_2").bulid(branch3x3_1)
        branch3x3_3 = IndiviualCNN(96,3,3,name = self.name+"inceptiona3x3_3").bulid(branch3x3_2)

        #branch4
        branchpool = AveragePooling2D((3, 3),strides=(1, 1),padding='same')(inputmodel)
        branchpool_1 = IndiviualCNN(32,1,1,name = self.name+"branchpool1").bulid(branchpool)

        output = [branch5x5_2,branch5x5_2,branch3x3_3,branchpool_1]
        #concancate
        IncA = concatenate(output,axis=3,name = self.name)
        return IncA

