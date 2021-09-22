# Overview

HipoMap is slide-based histopathology analysis framework in which a disease-specific graphical representation map is
generated from each slide. Further, HipoMap, which is a small and fixed size, is introduced as input for
machine-learning models instead of extremely large and variable size WSI. HipoMap is obtained using gradients of patch
probability scores to represent disease-specific morphological patterns. Proposed HipoMap based whole slide analysis has
outperformed current state-of-art whole slide analysis methods. We assessed the proposed method on Lung Cancer WSI
images and interpreted the model based on class probability scores and HipoMap scores. A pathologist clinically verified
the results of interpretation.

<img alt="HipoMap" src="../_images/hipomap.png">
