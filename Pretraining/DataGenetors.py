
import os
from os import walk
# from Datageneator import data_generator
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def ImgDataParameters(Rescale = 1. / 255,Shear_Range = 0.0,Zoom_Range = 0.0,Horizontal_Flip = True):
    datagen = ImageDataGenerator(
        rescale = Rescale,
        shear_range = Shear_Range,
        zoom_range = Zoom_Range,
        horizontal_flip = Horizontal_Flip)
    return datagen
def DataGenerator(inputdir,datatype,datagen,targetsize=(256, 256),colormode = "grayscale",batchsize = 32, classmode = 'binary', Seed = 42, Shuffle = True):
        data_generator = datagen.flow_from_director(
        directory=r"./%s/%s/"%(datatype,inputdir),
        target_size=targetsize,
        color_mode = colormode,
        batch_size = batchsize,
        class_mode = classmode,
        shuffle = Shuffle,
         seed = Seed
                         )
        Step_size = data_generator.n//data_generator.batch_size
        return data_generator, step_size
