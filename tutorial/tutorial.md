---
title: Deep Learning in 5 Minutes
author: sandbox-team
tutorial-id: XXX
experience: Beginner
persona: Data Scientist
source: Hortonworks
use case: Example (Data Discovery)
technology: XXX
release: hdp-3.0.1
environment: Sandbox
product: HDP
series: XX > XX
---

# Object Detection In 5 Minutes

## Introduction

![buggy](assets/buggy.jpg)

During the 2018 [DataWorks summit](Link) Hortonworks showcased an autonomous car which was trained to follow markers along a race track. The point of the exercise is to showcase the power of a TensorFlow container managed by YARN along with GPU isolation for fast deployment of Deep Learning models. 

In this tutorial We will briefly explore how Deep Learning plays a role in autonomous vehicles by using a pre-trained model to identify objects in a given image. We will employ [FasterRCNN+InceptionResNetV2](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1) network trained on [Open Images V4](https://storage.googleapis.com/openimages/web/index.html) imported to the environment using TensorFlow hub. On the clip below we can observe the perspective of the Hortonworks car.

![car-vision](assets/car-vision.gif)



## Prerequisites

- Downloaded and deployed the [Hortonworks Data Platform (HDP)](https://hortonworks.com/downloads/#sandbox) Sandbox
- [TensorFlow on YARN](http://example.com/link/to/required/tutorial)

## Outline

- [Concepts](#concepts)
- [Environment Setup](#section-title-1)
- [Section Title 2](#section-title-2)
- [Summary](#summary)
- [Further Reading](#further-reading)
- [Appendix A: Troubleshoot](#appendix-a-troubleshoot)

## Concepts

### CNN Inference

In AI terms _Inference_ refers to the ability of a Neural Network to classify objects based on previously presented data. [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) are a special type of network that is most efficient when working with image type data. To run inference with a CCN is to classify an image or an object within an image by using previously viewed data.

![elephant](assets/elephant.jpg)

To the human brain this is an elephant, albeit a drawing of one but an elephant nonetheless, we know this because we are able to make associations with previous shapes we have seen before and infer what type of animal it is we are observing. By the same token the inception_resnet_v2 has been trained on 600 categories which enable it to recognize objects such as vehicles, humans, and even footwear. 

We will run CNN inferences to the images below and explore how difference lighting and weather conditions can affect the results of the inference.

![collage](assets/collage.jpg)

note that this python script is a modified version of [Google Colab Object Detection.](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb)

## Environment Setup

- **python3**

Once you have Python3.6 installed ensure you also have these dependencies:

~~~bash
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv 
~~~

~~~bash
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate

pip3 install numpy
pip3 install matplotlib
pip3 install tf-hub
pip3 install tensorflow
~~~

### Import The Data set

Now that our environment has all the dependencies required we can bring in images to be processed.

Download the python script and data set that we will use for this tutorial:

~~~bash
cd ~/Downloads

wget github.com/raw-pythoncode

unzip objectDetection.zip

cd object-detection
~~~

## Run The Object Detection Model

To execute the python script that will execute the inference run:

~~~bash
source ~/venv/bin/activate
cd ~/Downloads/objectDetection/
~~~

~~~bash
python3 objectDetectionLocal.py --idir ./images/ --odir ./output/ --type jpg
~~~



## Summary

On the [CNN Transfer Learning Tutorial](James-tutorial)

## Further Reading

- [Google Colab](https://example.com)
- [Object Detection](https://hortonworks.com)
- [TensorFlow on YARN](https://hortonworks.com/blog/distributed-tensorflow-assembly-hadoop-yarn/)
- [TensorFlow Documentation](tf.com)
- [TensorFlow Hub Documentation](tfhub.com)

### Appendix A: Detect objects with your own dataset

In order to use the object detection model on your own images use the existing python script and 