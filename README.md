# HipoMap

![](Capture.PNG)

## Installation
HipoMap support `Python 3`.
It requires `numpy`, `pandas`, `tensorflow`, `scipy`, `scikit-learn`, `seaborn`, `matplotlib`, `openslide-python`, `cv2`.

### Quick installation of HipoMap

* Update system for install openslide-tools (to install openslide-python)
```
sudo apt-get update
```
* Install openslide-tools
```
sudo apt-get install openslide-tools
```
* Install openslide-python
```
pip install openslide
```
* Install HipoMap
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

#if you want to loaded your pre-trained model(.h5 file)
from tensorflow.keras.models import load_model 
model = load_model(r'./pre_model.h5')

#make representation map
from HipoMap.hipoMap import generateHipoMap
generateHipoMap(inputpath="/home/user/Dataset/", outputpath="/home/user/Rep/", model = model, layer_name="block5_conv3", patch_size=(224, 224))

#draw heatmap
from HipoMap.hipoMap import draw_represent
draw_represent(path="/home/yeon/Dataset/", K=50, max_value=1000, save=False)

#Classify data to cancer/normal with representation map
from HipoMap.hipoClassify import HipoClass
hipo = HipoClass(K=50)

#1. split data with base(.csv) 
trainset, validset, testset = hipo.split("./split.csv", dir_normal="/home/user/Dataset/Normal/", dir_cancer="/home/user/Dataset/Cancer")

#2. train the classifier
hipo_model = hipo.fit(trainset, validset, lr=0.1, epoch=20, batchsize=1, activation_size=196)

#3. get prediction value
prediction = hipo.predict(test_X=testset[0])

#4. get score (tpr, fpr, auc)
tpr, mean_fpr, auc = hipo.evaluate_score(label=testset[1], prediction=prediction)
```