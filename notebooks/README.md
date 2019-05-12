[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

This project was completed as the data scientist capstone requirement for the Udacity data scientist nanodegree. This folder consists of the completed Jupyter notebook for dog-project, which forked to my [github](https://github.com/jeremysmith244/dog_project).

Please follow instructions below in order to run the dog_app.ipnyb. In order to run this you will need to make sure you have correct requirements, and then you'll need to download the bottleneck features for the VGG16 and VGG19 models, along with haarcascade classifier which is part of cv2. Finally you will need to download the set of dog images, if you want to train the models yourself.

In addition to the .ipynb, all the functions required to detect dogs and humans in the dog_detector.py file. This file gets called by the predict_photo.py file, which operates like:

python predict_photo path_to_image

and prints the results of the VGG19 classifier to the terminal.

A note on the data. Some of the images (4 total I believe), were not being imported correctly by the keras data loader. Rather than fix the data loader, the import functions were simply updated to ignore images that could not be loaded.

## Project Instructions

### Instructions

0. Check requirements.txt

1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `dogImages`.

2. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.

3. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) as well as the [VGG-19 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) for the dog dataset.  Place in the repo, at location `bottleneck_features`.

4. Download the haarcascade xml file entitled 'harrcascade_frontalface_alt.xml' from [github](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in the location `haarcascade`.
