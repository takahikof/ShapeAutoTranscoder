# ShapeAutoTranscoder
## Introduction
This repository provides the code of ShapeAutoTranscoder (SAT) for unsupervised learning of 3D shape feature proposed in the following paper.<br>

"Transcoding across 3D Shape Representations for Unsupervised Learning of 3D Shape Feature"<br>
Takahiko Furuya and Ryutarou Ohbuchi<br>
(currently under review)<br>

SAT learns 3D shape feature via cross-predicting, or transcoding, multiple different 3D shape representations such as voxels, 3D point set, and multi-view 2D images. The embedded features in the latent space shared by the multiple 3D shape representations are used for 3D shape retrieval or 3D shape classification.
![pic](SAT.png)<br>
The instruction below describes how to prepare data and how to train/evaluate SAT using ModelNet10 dataset.

## Data
1. Download ModelNet10.zip from [the project webpage of ModelNet](https://modelnet.cs.princeton.edu/) and unzip the downloaded file.

## Pre-requisites:
Our code has been tested with Python 3.6, Tensorflow 1.13 and CUDA 10.0 on Ubuntu 18.04.<br>
Python packages required to run the code can be installed by using the command below. <br>
```
pip install tensorflow-gpu==1.13.2 scikit-learn Pillow h5py sobol sobol_seq pyrr moderngl
```
* Python
* CUDA
* TensorFlow
* h5py
* numpy
* scikit-learn
* sobol
* moderngl
* pyrr
* PIL

