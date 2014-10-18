# Kaggle CIFAR-10

Code for CIFAR-10 competition. http://www.kaggle.com/c/cifar-10

## Summary
|                   | Description                                                                            |
|-------------------|----------------------------------------------------------------------------------------|
| Data Augmentation | cropping, horizontal reflection [3] and scaling. see lib/data_augmentation.lua             |
| Preprocessing     | Global Contrast Normalization (GCN) and ZCA whitening. see lib/preprocessing.lua       |
| Model             | Very Deep Convolutional Networks with 3x3 kernel [1] |
| Training Time     | 20 hours on GTX760. |
| Prediction Time   | 2.5 hours on GTX760. |
| Result            | 0.93320 (single model). 0.94150 (average 6 models)|


## Neural Network Conﬁgurations

| Layer type       | Parameters                                |
|------------------|-------------------------------------------|
| input            | size: 24x24, channel: 3                   |
| convolution      | kernel: 3x3, channel: 64, padding: 1      |
| relu             |                                           |
| convolution      | kernel: 3x3, channel: 64, padding: 1      |
| relu             |                                           |
| max pooling      | kernel: 2x2, stride: 2                    |
| dropout          | rate: 0.25                                |
| convolution      | kernel: 3x3, channel: 128, padding: 1     |
| relu             |                                           |
| convolution      | kernel: 3x3, channel: 128, padding: 1     |
| relu             |                                           |
| max pooling      | kernel: 2x2, stride: 2                    |
| dropout          | rate: 0.25                                |
| convolution      | kernel: 3x3, channel: 256, padding: 1     |
| relu             |                                           |
| convolution      | kernel: 3x3, channel: 256, padding: 1     |
| relu             |                                           |
| convolution      | kernel: 3x3, channel: 256, padding: 1     |
| relu             |                                           |
| convolution      | kernel: 3x3, channel: 256, padding: 1     |
| relu             |                                           |
| max pooling      | kernel: 2x2, stride: 2                    |
| dropout          | rate: 0.25                                |
| fully connected  | channel: 1024                             |
| relu             |                                           |
| dropout          | rate: 0.5                                 |
| fully connected  | channel: 1024                             |
| relu             |                                           |
| dropout          | rate: 0.5                                 |
| softmax          | output: 10                                |

## Developer Environment

- Ubuntu 14.04
- 15GB RAM (This codebase can run on g2.2xlarge!)
- CUDA (GTX760 or more higher GPU)
- [Torch7](http://torch.ch/) latest
- [cuda-convnet2.torch](https://github.com/soumith/cuda-convnet2.torch)

## Installation

Install CUDA (on Ubuntu 14.04):

    apt-get install nvidia-331
    apt-get install nvidia-cuda-toolkit

Install Torch7 (see [Torch (easy) install](https://github.com/torch/ezinstall)):

    curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash

Install(or upgrade) dependency packages:

    luarocks install torch
    luarocks install nn
    luarocks install cutorch
    luarocks install cunn
    luarocks install https://raw.githubusercontent.com/soumith/cuda-convnet2.torch/master/ccn2-scm-1.rockspec

### Checking CUDA environment

    th cuda_test.lua

Please check your Torch7/CUDA environment when this code fails.

### Convert dataset

Place the [data files](http://www.kaggle.com/c/cifar-10/data) into a subfolder ./data.

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

## MISC

### Model Averaging

Training with different seed parameter for each nodes.

    th train.lua -seed 11
    th train.lua -seed 12
    ...
    th train.lua -seed 16

Mount the `models` directory for each nodes. for example, `ec2/node1`, `ec2/node2`, .., `ec2/node6`.

Edit the path of model file in `predict_averaging.lua`.

Run the prediction command.

    th predict_averaging.lua

### Network In Network

`./nin_model.lua` is an implementation of Network In Network [2].
This model gives score of 0.92400.

## Figure

data augmentation + preprocessing

![data-augmentation-preprocessing](https://raw.githubusercontent.com/nagadomi/kaggle-cifar10-torch7/master/figure/zca.png)

## References
- [1] Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", [link](http://arxiv.org/abs/1409.1556)
- [2] Min Lin, Qiang Chen, Shuicheng Yan, "Network In Network", [link](http://arxiv.org/abs/1312.4400)
- [3] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", [link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [4] R. Collobert, K. Kavukcuoglu, C. Farabet, "Torch7: A Matlab-like Environment for Machine Learning"
