# BFAct: Out-of-distribution Detection with Butterworth Rectified Activations

This is the source code for the paper [link to paper] by Haoan Li and Haojia Kong

In this work, we propose BFAct for reducing model overconfidence on OOD data.

## Usage

### 1. Prepare Datasets

The datasets for this program is placed in the folder `./data`. Some of the datasets are downloaded automatically in runtime, while others needed to be downloaded in prior to execution.

#### ID dataset

The ImageNet dataset can be downloaded from here [Imagenet-1k ](http://www.image-net.org/challenges/LSVRC/2012/index)

The [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) datasets are downloaded automatically

#### OOD datasets

Four of the OOD datasets are taken from [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), [Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf).

They can be downloaded through the following links

```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

And [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/) from their original site.

### 2. Models

The model that can be used with this program is ResNet18, ResNet50, and MobileNet_v2 from PyTorch. They are downloaded upon first execution. 

If you would like to run other datasets with the program, it is necessary to train your own models.

### 3. OOD Detection

To replicate the result, it suffice to run the following command in the command line:

```bash
python main.py --threshold 1.6
```

For help on the available commands, please refer to:

```bash
python main.py -h
```
