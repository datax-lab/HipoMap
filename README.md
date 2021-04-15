# HipoMap

![](Capture.PNG)

## Installation
HipoMap support `Python 3`.
It requires `numpy`, `pandas`, `tensorflow`, `scipy`, `scikit-learn`, `seaborn`, `matplotlib`, `openslide`.

```
pip install hipo_map
```

## Documentation

## Quick Start
```python
#Model load
#if you want to loaded keras pre-trained model
from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16()

#if you want to loaded your pre-trained model
from tensorflow.keras.models import load_model 
model = model = load_model(r'./pre_model.h5')

#make representation map
from hipo_map.Hipomap import generateHipoMap
generateHipoMap(inputpath="/home/user/Dataset/", outputpath="/home/user/Rep/", model = model, layer_name="block5_conv3", patch_size=(224, 224))

#draw heatmap
from hipo_map.Hipomap import  draw_represent
draw_represent(path="/home/yeon/Dataset/", K=50, max_value=1000)

#Classify data to cancer/normal with representation map
from hipo_map.hipoClassify import HipoMap
hipomap = HipoMap(K=50)

#1. split data with base(.csv) 
trainset, validset, testset = hipomap.split("./split.csv", dir_normal="/home/user/Dataset/Normal/", dir_cancer="/home/user/Dataset/Cancer")

#2. train the classifier
hipo_model = hipomap.fit(trainset, validset, lr=0.1, epoch=20, batchsize=1, activation_size=196)

#3. get prediction value
prediction = hipomap.predict(test_X=testset[0])

#4. get score (tpr, fpr, auc)
tpr, mean_fpr, auc = hipomap.evaluate_score(label=testset[1])
```