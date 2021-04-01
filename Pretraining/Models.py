import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import models
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import applications
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
# from keras.applications.inception_v3 import CATNet2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from efficientnet.keras import EfficientNetB3
def efficientNetB3model():
    base_model = EfficientNetB3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
def inceptionv3_model():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
def Resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
def Densenet_model():
    base_model = DenseNet121(weights= None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def CATNet21():

    base_model = InceptionV3(include_top=False, weights=None,input_shape = (299,299,3))
    base_model1 = InceptionV3(include_top=False, weights=None,input_shape = (299,299,3))
    for l in base_model1.layers:
        

        l.name = "%s_workaround" % l.name
    x = base_model.output
    
    x = GlobalAveragePooling2D()(x)
    y =  base_model1.output
    y = GlobalAveragePooling2D()(y)

    combined = layers.concatenate([x, y])
    z = Dense(1024, activation='relu')(combined)
    predictions = Dense(1, activation='softmax')(z)
    model = Model(inputs=[base_model.input,base_model1.input], outputs=predictions)

    return model


def DoTNetmodel():
    input_img = Input(shape=(299, 299, 1))
    model1 = Conv2D(50, kernel_size=(3, 3), activation='tanh', dilation_rate=(2, 2), padding='valid')(input_img)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    model1 = Dropout(0.1)(model1)
    model1 = Conv2D(50, kernel_size=(3, 3), activation='tanh', dilation_rate=(2, 2), padding='valid')(model1)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    model1 = Dropout(0.1)(model1)
    model1 = Conv2D(50, kernel_size=(3, 3), activation='tanh', dilation_rate=(2, 2), padding='valid')(model1)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    model1 = Dropout(0.1)(model1)
    outmodelf = Flatten()(model1)
    model = Dense(units=50, activation='tanh', input_dim=50, kernel_initializer='uniform')(outmodelf)
    model = Dense(units=1, activation='softmax', kernel_initializer='uniform')(model)
    model = Model(input_img, model)
    return model

def Vgg191():
    base_model = VGG16(weights=None, include_top=False,input_shape=(299,299,3))

    # add a global spatial average pooling layer
    print(base_model.summary())

    # x = base_model['block5_conv3'].output
    x = base_model.output
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    # let's add a fully-connected layer
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes

    predictions = Dense(1, activation='sigmoid')(x)


    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
