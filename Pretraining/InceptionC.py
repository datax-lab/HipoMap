from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from tensorflow.keras.layers import  BatchNormalization,Activation,Dropout,SpatialDropout2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D,AveragePooling2D,concatenate
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
    
        Dilatedlayer = Conv2D(self.filters, (self.num_row,self.num_col),dilation_rate = self.dilation_rate,strides=self.strides,padding=self.Padding,use_bias=False,name = self.conv_name,  kernel_regularizer = self.Kernel_regularizer)(inputlayer)
        BatchNor = BatchNormalization(axis = 1, scale= False, name = self.bn_name)(Dilatedlayer)
        BatchNor = SpatialDropout2D(0.4)(BatchNor)
        Act = Activation('relu', name= self.name)(BatchNor)
        return Act
        

class InceptionC():
    
    def __init__(self, name,c7):
        
        self.name = name
        self.c7 = c7

        

    def bulid(self,inputmodel):
        #branch1
        branch1x1 = IndiviualCNN(192,1,1,name = self.name+"inceptionc1x1").bulid(inputmodel)

        branch7x7_b1 = IndiviualCNN(self.c7,1,1,name = self.name+"inceptionc7x7c_1").bulid(inputmodel)
        branch7x7_b2 = IndiviualCNN(self.c7,7,1,name = self.name+"inceptionc7x7c_2").bulid(branch7x7_b1)
        branch7x7_b3 = IndiviualCNN(self.c7,1,7,name = self.name+"inceptionc7x7c_3").bulid(branch7x7_b2)
        branch7x7_b4 = IndiviualCNN(self.c7,7,1,name = self.name+"inceptionc7x7c_4").bulid(branch7x7_b3)
        branch7x7_b5 = IndiviualCNN(192,1,7,name = self.name+"inceptionc7x7c_5").bulid(branch7x7_b4)

        branchpool = AveragePooling2D((3, 3),strides=(1, 1),padding='same')(inputmodel)
        branchpool_1 = IndiviualCNN(192,1,1,name = self.name+"branchpool1").bulid(branchpool)

        output = [branch1x1,branch7x7_b5,branchpool_1 ]

        IncB = concatenate(output,axis=3,name = self.name)
        return IncB
