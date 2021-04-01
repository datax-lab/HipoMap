import pandas as pd
from tensorflow.keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras import optimizers
import os
import random
from sklearn.model_selection import train_test_split,KFold
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
#random.seed(42)
import os
from os import walk
# from Datageneator import data_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.callbacks import EarlyStopping, ModelCheckpoint
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
# from Models import inceptionv3_model,DoTNetmodel,Vgg19
from DataGenetors import ImgDataParameters,DataGenerator
from CAT_Net2 import CATNet2
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import multi_gpu_model
inputdir = "/home/wsai/Dataset"
# ##Data_Genetor
'''
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True)
#
train_generator = datagen.flow_from_directory(
    directory=r"/home/skosaraju/Expirementsdata/Train_ck_pd",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42
)

valid_generator = datagen.flow_from_directory(
    directory=r"/home/skosaraju/Expirementsdata/Valid_ck_pd",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=1,
    class_mode='binary',
    shuffle=True,
    seed=42
)
#
test_generator = datagen.flow_from_directory(
    directory=r"/home/skosaraju/Expirementsdata/Test_ck_pd",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=1,
    class_mode='binary',
    shuffle=False,
    seed=42
)
'''
input_imgen = ImageDataGenerator(rescale = 1./255 )


test_imgen = ImageDataGenerator(rescale = 1./255)



def generate_generator_multiple(generator,dir1, dir2):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (299,299),
                                          class_mode = 'binary',
                                          color_mode="grayscale",
                                          batch_size = 4,
                                          shuffle=False, 
                                          seed=42)
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (299,299),
                                          class_mode = 'binary',
                                          color_mode="grayscale",
                                          batch_size = 4,
                                          shuffle=False, 
                                          seed=42)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label
            
            
traingenerator=generate_generator_multiple(generator=input_imgen,
                                           dir1= r"/home/wsai/Dataset/20x/Train",
                                           dir2= r"/home/wsai/Dataset/5x/Train",
                                          )
     

validationgenerator = generate_generator_multiple(input_imgen,
                                          dir1= r"/home/wsai/Dataset/20x/Valid",
                                          dir2= r"/home/wsai/Dataset/5x/Valid",
                                          )
     
testgenerator=generate_generator_multiple(test_imgen,
                                            dir1= r"/home/wsai/Dataset/20x/Test",
                                            dir2= r"/home/wsai/Dataset/5x/Test",
                                                                                    )
          


batch_size = 4
# print([train_generator.__getitem__(0)[0]])
Lr = [0.00001]
# Lr = [0.001]
Beta_1 = [0.85]
# Beta_1 = [0.85,0.9]
Paramaters_list = []
for lr in  Lr:
    for beta1 in Beta_1:
        Adam1 = optimizers.Adam(lr=lr , beta_1 = beta1)
        InputA = Input(shape=(299, 299,1))
        InputB = Input(shape=(299, 299,1))
        model = CATNet2.bulid(InputA, InputB)
        model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer = Adam1, loss = 'mean_squared_error',metrics=['accuracy'])
        epochs = 50
        model.fit_generator(traingenerator,
                        steps_per_epoch=int(90995/4),
                        epochs = epochs,

                        use_multiprocessing=False,
                        shuffle=False)

        model.save('ModelInterpretationCat_Net2.h5')

