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
        self.Kernel_regularizer = l2(0.04)

        
        
    def bulid(self,inputlayer):
    
        Dilatedlayer = Conv2D(self.filters, (self.num_row,self.num_col),dilation_rate = self.dilation_rate,
                              strides=self.strides,padding=self.Padding,use_bias=False,name = self.conv_name, kernel_regularizer = self.Kernel_regularizer)(inputlayer)
        BatchNor = BatchNormalization(axis = 1, scale= False, name = self.bn_name)(Dilatedlayer)
        BatchNor = SpatialDropout2D(0.4)(BatchNor)
        Act = Activation('tanh', name= self.name)(BatchNor)

        return Act
        

class InceptionE():
    
    def __init__(self, name):
        
        self.name = name

        
        

    def bulid(self,inputmodel):
        #branch1
        branch1x1_1 = IndiviualCNN(320,1,1,name = self.name+"inceptione1x1").bulid(inputmodel)

        
        #branch2
        branch3x3_1 = IndiviualCNN(384,1,1,name = self.name+"inceptione3x3_1").bulid(inputmodel)
        branch3x3_2a = IndiviualCNN(384,1,3,name = self.name+"inceptione3x3_2a").bulid(branch3x3_1)
        branch3x3_2b = IndiviualCNN(384,3,1,name = self.name+"inceptione3x3_2b").bulid(branch3x3_1)
        branch3x3 = concatenate([branch3x3_2a, branch3x3_2b], axis=3, name = self.name+"inceptione3x3_concat1")
        
        branch3x3_2 = IndiviualCNN(448,1,1,name = self.name+"inceptione3x3_2").bulid(inputmodel)
        branch3x3_3 = IndiviualCNN(384,1,1,name = self.name+"inceptione3x3_3").bulid(branch3x3_2)
        branch3x3_4a = IndiviualCNN(384,1,3,name = self.name+"inceptione3x3_4a").bulid(branch3x3_3)
        branch3x3_4b = IndiviualCNN(384,3,1,name = self.name+"inceptione3x3_4b").bulid(branch3x3_3)
        branch3x3_5 = concatenate([branch3x3_4a, branch3x3_4b], axis=3, name =self.name+ "inceptione3x3_concat2")
        

        
        #branch3
        branchpool = AveragePooling2D((3, 3),strides=(1, 1),padding='same')(inputmodel)
        branchpool_1 = IndiviualCNN(192,1,1,name = self.name+"incewptionebranchpool1").bulid(branchpool)

        
        output = [branch1x1_1,branch3x3 ,branch3x3_5, branchpool_1]
        #concancate
        IncB = concatenate(output,axis=3,name = self.name)
        return IncB
