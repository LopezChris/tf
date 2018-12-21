---
title: Object Detection in 5 Minutes
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

During the 2018 [DataWorks summit](Link) Hortonworks showcased an autonomous car which was trained to follow markers along a race track only by using images as a reference.

The point of the exercise is to showcase the power of a TensorFlow container managed by YARN along with GPU isolation for fast deployment of Deep Learning models.

On the clip below we can observe the perspective of the Hortonworks car.

![car-vision](assets/car-vision.gif)

In this tutorial we will briefly explore Deep Learning plays a role in autonomous vehicles by using a pre-trained model to identify objects in a given image. We will employ [FasterRCNN+InceptionResNetV2](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1) network trained on [Open Images V4](https://storage.googleapis.com/openimages/web/index.html) imported to the environment using TensorFlow Hub. Originally the [InceptionResNetV2](https://www.tensorflow.org/versions/r1.6/api_docs/python/tf/keras/applications/InceptionResNetV2) feature extractor was trained on [ImageNet](http://www.image-net.org/) and combined with [FasterRCNN](https://arxiv.org/abs/1506.01497) on [OpenImages V4 dataset](https://storage.googleapis.com/openimages/web/index.html)

## Prerequisites

- Downloaded and deployed the [Hortonworks Data Platform (HDP)](https://hortonworks.com/downloads/#sandbox) Sandbox
- [TensorFlow on YARN](http://example.com/link/to/required/tutorial)

## Outline

- [Concepts](#concepts)
- [Environment Setup](#environment-setup)
- [Import the Data Set](#import-the-data-set)
- [Summary](#summary)
- [Further Reading](#further-reading)
- [Appendix A: Troubleshoot](#appendix-a-troubleshoot)

## Concepts

### CNN Inference

In AI terms _Inference_ refers to the ability of a Neural Network to classify objects based on previously presented data. [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) are a special type of network that is most efficient when working with image type data. To run inference with a CCN is to classify an image or an object within an image by using previously viewed data

![elephant](assets/elephant.jpg)

To the human brain this is an image of an elephant, albeit a drawing of one but an elephant nonetheless, we know this because we are able to make associations with previous shapes we have seen before and infer what type of animal we are observing. By the same token the inception_resnet_v2 has been trained on 600 categories which enable it to recognize objects such as vehicles, humans, elephants, and even footwear. The inferences we run in this tutorial are generalized; however, it will show you 

We will run CNN inferences to the images below and explore how difference lighting and weather conditions can affect the results of the inference.

![collage](assets/collage.jpg)

> NOTE: The python script used is a modified version of [Google Colab Object Detection.](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb)

## Environment Setup

You will need the following libraries installed:

- **python3**
- **numpy**
- **matplotlib**
- **tf-hub**
- **tensorflow**

~~~bash
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv 
~~~

~~~bash
virtualenv --system-site-packages -p python3 ./venv # From home direcotry
source ./venv/bin/activate

pip3 install numpy
pip3 install matplotlib
pip3 install tensorflow
pip3 install tf-hub
~~~

### Import The Data Set

Now that our environment has all the dependencies required we can bring in images to be processed.

Download the python script and data set that we will use for this tutorial:

~~~bash
cd ~/Downloads

wget github.com/raw-pythoncode

unzip objectDetection.zip

cd object-detection
~~~

## Run The Object Detection Model

To execute the python script that will execute the inference on the sample images first ensure that your virtual environment is active:

~~~bash
source ~/venv/bin/activate
cd ~/Downloads/objectDetection/
~~~

next execute the python script on the folder you've just downloaded

~~~bash
python3 objectDetectionLocal.py --idir ~/Downloads/object-detection/images/ --odir ~/Downloads/object-detection/output/ --type jpg
~~~

## Results

The first image is of a man on the sidewalk of a busy street, as we can see the lighting condition is good and the focus is sharp on the subject of the picture, these conditions allow the model to infer on the objects in the picture with relatively high precision.

![man-waling-output](assets/man-waling-output.jpg)

The next image depicts a typical 

![good-lighting-traffic-output](assets/good-lighting-traffic-output.jpg)

Is

![good-lighting-traffic-output](assets/good-lighting-traffic2-output.jpg)

The next image represents non-ideal conditions where the objects in the picture are

![bad-condition-output](assets/bad-condition-output.jpg)

The last image in the set is of 

![bad-condition-output](assets/bad-condition2-output.jpg)

## Summary

Congratulations, now you know how to run inference using a pre-trained model. Naturally, you might want to detect custom objects using your own data, on the [CNN Transfer Learning Tutorial](James-tutorial) you will learn to do just that by first training a model to identify elephants.

## Further Reading

- [Google Colab](https://example.com)
- [Object Detection](https://hortonworks.com)
- [TensorFlow on YARN](https://hortonworks.com/blog/distributed-tensorflow-assembly-hadoop-yarn/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/)
- [TensorFlow Hub Documentation](https://www.tensorflow.org/hub/api_docs/python/hub)

### Appendix A: Detect objects with your own dataset

Because the model has been pretrained on 600 categories it can be really fun to take images of your own and observe the output, and once you have the python script on your {SB} running inference on your own images is fairly simple.

Simply change the parameters in the python script in the `object-detection` folder we unzipped earlier:

~~~bash
python3 objectDetectionLocal.py --idir {path_to_input_directory} --odir {path_to_output_directory} --type {image_type (e.g. jpg, png, jpeg)}
~~~