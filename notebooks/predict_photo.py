from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from extract_bottleneck_features import extract_VGG19
import pickle
import numpy as np
import cv2
from dog_detector import *
import argparse

# Define the keras model to do prediction on top of pretrained VGG19
VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
VGG19_model.add(Dense(1000, activation='relu'))
VGG19_model.add(Dropout(0.5))
VGG19_model.add(Dense(133, activation='softmax'))

VGG19_model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

# Load weights for the model
VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')

# Load the dictionary of category names
with open ('dog_names', 'rb') as fp:
    dog_names = pickle.load(fp)


def path_to_tensor(img_path):
    '''Loads an image form a path, converts to tensor for NN input

    INPUT:
    img_path - location of Image

    OUPUT:
    tensor - the converted data suitable for keras input'''

    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def VGG19_predict_breed(img_path):
    '''Using VGG19 based convolution network, predict breed form image path

    INPUT:
    img_path - location of the Image

    OUPUT:
    breed - string with the breed most closly matching image'''

    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def human_or_dog(img_path):
    '''Predict dog or human and if either predict closest dog breed

    INPUT:
    img_path - location of the Image

    OUPUT:
    tuple - (dog or human or False, breed or False)'''

    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if dog_detector(img_path):
        print('hello doggie!')
        breed = VGG19_predict_breed(img_path)
        print('I think yous a {}'.format(breed))
        return ('dog', breed)

    if face_detector(img_path):
        print('hello person!')
        breed = VGG19_predict_breed(img_path)
        print('you look like a {}'.format(breed))
        return ('human', breed)

    else:
        print('I dont know what you are')
        return(False, False)


parser = argparse.ArgumentParser(
    description='Required image path',
    )

# Collect the inputs needed with argparse
parser.add_argument('img_path', action='store', help='Location of image on\
                     which to run prediction.')

inpts = parser.parse_args()
img_path = inpts.img_path

human_or_dog(img_path)
