# Datascientist Capstone Project

Welcome to my capstone project for Udacity Datascientist nanodegree.

This project consists of two components. This first folder, notebooks, contains all of the model architecture, training of the models and discussion of the machine learning methods and accuracy of the test datasets. Users interested in reading through the model training portion, or interested in training there own models, should read through this section. All code is covered under the license there, and is from [Udacity](https://www.udacity.com/).

The second component is a Flask based web app, which given a train model output after following the notebook, will allow a user to upload an image to a locally hosted webpage and run the dog breed classifier on it. The page will first check if the image seems to be a dog or a human, and if it is not it will just say that it does not know what this is. If it is either a dog or a human, it will output a prediction as to the closest breed that it thinks you look like. On a set of thousands of dog images, the ML algorithm was ~72% accurate for predicting dog breeds, and 100% fun for humans to try to play with! All code from this section is modified form this [excellent template](https://github.com/mtobeiyf/keras-flask-deploy-webapp).
