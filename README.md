# Kaggle CIFAR-10

Code for CIFAR-10 competition. http://www.kaggle.com/c/cifar-10

## Summary
|                   | Description                                                                            |
|-------------------|----------------------------------------------------------------------------------------|
| Data Augmentation | cropping, scaling and horizontal reflection. see lib/data_augmentation.lua             |
| Preprocessing     | Global Contrast Normalization (GCN) and ZCA whitening. see lib/preprocessing.lua       |
| Model             | Network In Network (NIN). see nin_model.lua |
| Training Time     | 15 hours on GTX760. |
| Prediction Time   | 5 hours on GTX760. |
| Result            | 0.91730 in public leaderboard. |

## Developer Environment

- Ubuntu 14.04
- LuaJit/Torch7 latest
- 32GB RAM
- CUDA environment (GTX760 or more higher GPU)

## Installation

Install CUDA (on Ubuntu 14.04):

    apt-get install nvidia-331
    apt-get install nvidia-cuda-toolkit

Install Torch7:

    curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash

Install(or upgrade) dependency packages:

    luarocks install torch
    luarocks install nn
    luarocks install cutorch
    luarocks install cunn

### Checking CUDA environment

    th cuda_test.lua

Please check your Torch7/CUDA environment when this code fails.

### Convert dataset

Please place the [data files](http://www.kaggle.com/c/cifar-10/data) into a subfolder ./data.

    ls ./data
    test  train  trainLabels.csv
-
    th convert_data.lua

## Local testing

    th validate.lua

dataset:

| train   | test        |
| ------- | ----------- |
| 1-40000 | 40001-50000 |

## Generating the submission.txt

    th train.lua
    th predict.lua

## Figure
data augmentation + preprocessing

![data-augmentation-preprocessing](https://raw.githubusercontent.com/nagadomi/kaggle-cifar10-torch7/master/figure/zca.png)

## References
- Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks"
- Min Lin, Qiang Chen, Shuicheng Yan, "Network In Network"
