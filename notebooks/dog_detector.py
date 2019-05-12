from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
import numpy as np
import cv2

def path_to_tensor(img_path):
    '''Loads an image form a path, converts to tensor for NN input'''

    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return
    return np.expand_dims(x, axis=0)


def ResNet50_predict_labels(img_path):
    '''Uses pretrained ResNet50 to classify images

    INPUT:
    img_path - location of images

    OUPUT:
    labels - the prediction of ResNet50 on the image
    '''

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    labels = np.argmax(ResNet50_model.predict(img))

    return labels

def dog_detector(img_path):
    '''Runs ResNet50 predictions, boolean true returns if dog is found'''

    prediction = ResNet50_predict_labels(img_path)
    dog_found = ((prediction <= 268) & (prediction >= 151))

    return dog_found


def face_detector(img_path):
    '''Uses cv2 inbuilt haarcascade human face detector to find faces

    INPUT:
    img_path - location of Image

    OUPUT:
    found_face - boolean True if more than 0 faces found'''

    face_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_alt.xml')

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
