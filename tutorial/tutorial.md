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

# TODO:

Update metadata

# Deep Learning In 5 Minutes

## Introduction

Convolutional Neural Networks are [insert knowledge here]() and are used for [Drop Knowledge here]().

![elephant](assets/elephant.jpg)

To the human brain this is an elephant, albeit a fake on but an elephant nonetheless, we know this because we are able to make associations with previous shapes we have seen before and infer what type of animal it is we are observing.
Similarly in this tutorial we will explore how a computer envisions the world around it, at least when it has been trained.

## Prerequisites

- Downloaded and deployed the [Hortonworks Data Platform (HDP)](https://hortonworks.com/downloads/#sandbox) Sandbox
- [TensorFlow on YARN](http://example.com/link/to/required/tutorial)

## Outline

- [Concepts](#concepts)
- [Section Title 1](#section-title-1)
- [Section Title 2](#section-title-2)
- [Summary](#summary)
- [Further Reading](#further-reading)
- [Appendix A: Troubleshoot](#appendix-a-troubleshoot)
- [Appendix B: Extra Features](#appendix-b-extra-features)

> Note: It may be good idea for each procedure to start with an action verb (e.g. Install, Create, Implement, etc.). Also note that for notes we use ">" to start a note. There are other ways to do it but this is our standard

## Concepts

Sometimes, it's a good idea to include a concepts section to go over core concepts before the real action begins.  That way:

- Readers get a preview of the tech that'll be introduced.
- The section can be used as a reference for terminology brought up throughout the tutorial.
- By the way, this is an example of a list.  Feel free to copy/paste this for your own use.
- Use a single space for lists, not tabs.
  - Also, a sub-list.

## Environment Setup

Have dependencies installed preferably in a virtual environment

~~~bash
pip3 install virtualenv
~~~

Make and activate your environment

~~~bash
mkdir ~/venvtf
virtualenv ~/venvtf
~~~

Source and activate

~~~bash
source ~/venvtf/bin/activate
~~~

Now for the good stuff

~~~bash
python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
~~~

Other Dependencies

~~~bash
brew install python3
pip3 install numpy
pip3 install tensorflow-hub
~~~

### Import Data

Now that our environment has all the dependencies required we can bring in images to be processed.

~~~bash
scp something something or wget idk yet
~~~

## Run Inference



## Summary

Congratulations, you've finished your first tutorial!  Including a review of the tutorial or tools they now know how to use would be helpful.

## Further Reading

- [Reference Article Title](https://example.com)
- [Title of Another Useful Tutorial](https://hortonworks.com)

> Note: A tutorial does not go as in depth as documentation, so we suggest you include links here to all documents you may have utilized to build this tutorial.

### Appendix A: Troubleshoot

The appendix covers optional components of the tutorial, including help sections that might come up that cover common issues.  Either include possible solutions to issues that may occur or point users to [helpful links](https://hortonworks.com) in case they run into problems.

### Appendix B: Extra Features

Include any other interesting features of the big data tool you are using.

Example: when learning to build a NiFi flow, we included the necessary steps required to process the data. NiFi also has additional features for adding labels to a flow to make it easier to follow.