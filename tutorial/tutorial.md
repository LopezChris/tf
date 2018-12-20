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

Convolutional Neural Networks (CNNs) are [insert knowledge here]() and are used for [Drop Knowledge here]().

![buggy](assets/buggy.jpg)

During the 2018 [DataWorks summit](Link) Hortonworks showcased an autonomous car which was trained to follow markers along a race track. The point of the exercise is to showcase the power of a TensorFlow container managed by YARN along with GPU isolation for fast deployment of Deep Learning models.

In this tutorial We will employ [FasterRCNN+InceptionResNetV2](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1) network trained on [Open Images V4](https://storage.googleapis.com/openimages/web/index.html) imported to the environment using TensorFlow hub

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

![elephant](assets/elephant.jpg)

To the human brain this is an elephant, albeit a drawing of one but an elephant nonetheless, we know this because we are able to make associations with previous shapes we have seen before and infer what type of animal it is we are observing. By the same token the inception_resnet_v2 has been trained on 
Similarly in this tutorial we will explore how a computer envisions the world around it, or at least on what it has been trained to look for.

## Environment Setup

Ensure that the environment you are using has virtualenv with:

- **python3**
- **numpy**
- **matplotlib**
- **tf-hub**
- **tensorflow**

Have dependencies installed in a virtual environment

~~~bash
pip3 install virtualenv
~~~

Make and activate your environment

~~~bash
mkdir ~/venv
virtualenv ~/venv
~~~

Source and activate

~~~bash
source ~/venv/bin/activate
~~~

> NOTE: You will need python 3.6 to use this tutorial

~~~bash
apt-get install python3
~~~

~~~bash
pip3 install numpy
pip3 install tensorflow-hub
~~~

### Import The Data set

Now that our environment has all the dependencies required we can bring in images to be processed.

Download the python script and data set that we will use for this tutorial

~~~bash
wget github.com/raw-pythoncode
~~~

~~~bash
unzip objectDetection.zip
~~~

note that this tutorial is modified from [Google Colab Object Detection.](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb)

Here are some of the images we've just downloaded:

![elephant](assets/elephant.jpg)

## Run The Object Detection Model

To execute the python script that will execute the inference run:

~~~bash
source ~/venv/bin/activate
cd ~/Downloads/objectDetection/
~~~

~~~bash
python3 objectDetectionLocal.py --idir ./images/ --odir ./output/ --type jpg
~~~

![elephant](assets/elephant.jpg)

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