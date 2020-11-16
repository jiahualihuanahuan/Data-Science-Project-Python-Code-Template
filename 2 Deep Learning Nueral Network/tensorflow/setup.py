# Using tensorflow backend
import keras

#-------------------------------------
# Using plaidml.keras.backend backend
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

#-------------------------------------
# Using cuda (NVDIA GPU) backend