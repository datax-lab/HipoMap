from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from tensorflow.keras.layers import  BatchNormalization,Activation,Dropout,SpatialDropout2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D,AveragePooling2D,concatenate
from keras.regularizers import l2

from keras.regularizers import l2
        
class IndiviualCNN():

    def __init__(self, filters,num_row, num_col,name,strides = (1,1)):
        self.strides = strides
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
    
        Dilatedlayer = Conv2D(self.filters, (self.num_row,self.num_col),dilation_rate = self.dilation_rate,strides=self.strides,
                              padding=self.Padding,use_bias=False,name = self.conv_name, kernel_regularizer = self.Kernel_regularizer)(inputlayer)
        BatchNor = BatchNormalization(axis = 1, scale= False, name = self.bn_name)(Dilatedlayer)
        BatchNor = SpatialDropout2D(0.4)(BatchNor)
        Act = Activation('relu', name= self.name)(BatchNor)
        return Act
        

class InceptionD():
    
    def __init__(self, name):
        
        self.name = name


    def bulid(self,inputmodel):
        #branch1
        branch3x3_1 = IndiviualCNN(192,1,1,name = self.name+"inceptiond3x3").bulid(inputmodel)
        branch3x3_2 = IndiviualCNN(320,3,3,name = self.name+"inceptiond13x3",strides = (2,2)).bulid(branch3x3_1)

        #branch2
        branch7x7_1 = IndiviualCNN(192,1,1,name = self.name+"inceptiond7x7_1").bulid(inputmodel)
        branch7x7_2 = IndiviualCNN(192,1,7,name = self.name+"inceptiond7x7_2").bulid(branch7x7_1)
        branch7x7_3 = IndiviualCNN(192,7,1,name = self.name+"inceptiond7x7_3").bulid(branch7x7_2)
        branch7x7_4 = IndiviualCNN(192,3,3,name = self.name+"inceptiond7x7_4",strides = (2,2)).bulid(branch7x7_3)

        #branch3
        branchpool = MaxPooling2D((3, 3),strides=(2, 2),padding='same')(inputmodel)

        output = [branch3x3_2,branch7x7_4,branchpool]
        #concancate
        IncB = concatenate(output,axis=3,name = self.name)
        return IncB
