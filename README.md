# TensorFlow Playground

![experiments/results/elephant.jpg](experiments/results/elephant.jpg)

## Object Detection With RC Car

![experiments/results/objectid.gif](experiments/results/objectid.gif)

That is me and my buddy James being recognized by artificial intelligence

## Get started

### This is my setup

It is not ideal but it works for proof of concept

~~~python
Memory      10.1 GiB
Processor   Intel® Core™ i7-4870HQ CPU @ 2.50GHz × 2
Graphics    llvmpipe (LLVM 6.0, 256 bits) (VM)
GNOME       3.28.2
OS Type     64-bit
Disk        111.3 GB
~~~

### Virtual Box Settings

Initially I had a very slow VM this is what made a difference:

~~~python
Video Memory            128 MB
Enable 3D acceleration  FALSE
~~~

## Libraries & Software

Ensure that the environment you are using has virtualenv with:

- **python3**
- **numpy**
- **matplotlib**
- **tf-hub**
- **tensorflow**

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
brew install python3
pip3 install numpy
pip3 install tensorflow-hub
~~~

## Run code

~~~bash
source ~/venv/bin/activate
python3 <NAME OF .py FILE>
~~~
