# Model load

# If you want to loaded keras pre-trained model
from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16()

# If you want to loaded your pre-trained model(.h5 file)
from tensorflow.keras.models import load_model

model = load_model(r'./pre_model.h5')

# Make representation map
from HipoMap.hipoMap import generateHipoMap

generateHipoMap(inputpath="<path>/Dataset/", outputpath="<path>/Rep/", model=model,
                layer_name="block5_conv3", patch_size=(224, 224))