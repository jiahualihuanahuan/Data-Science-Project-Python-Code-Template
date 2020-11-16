# Using plaidml.keras.backend backend
import os, shutil
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = 'E:/Dropbox/Machine Learning/Data/dogs-vs-cats/'

# The directory where we will
# store our smaller dataset
base_dir = 'E:/Dropbox/Machine Learning/Data/dogs-vs-cats/data/'
#os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
#os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
#os.mkdir(test_dogs_dir)

'''
# Copy first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(6250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(6250, 9375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(9375, 12500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(6250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(6250, 9375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(9375, 12500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
'''

# build a VGG16 neural network model
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

VGG_16 = Sequential()
VGG_16.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
VGG_16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
VGG_16.add(Flatten())
VGG_16.add(Dense(4096,activation="relu"))
VGG_16.add(Dense(4096,activation="relu"))
VGG_16.add(Dense(1, activation="softmax"))

VGG_16.summary()

# compile model
from keras import optimizers

VGG_16.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# prepare data
from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(224, 224),
        batch_size=56,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=56,
        class_mode='binary')

# check batch shape
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto', 
                             period=1)
early = EarlyStopping(monitor='val_acc', 
                      min_delta=0, 
                      patience=5, 
                      verbose=1, 
                      mode='auto')

# fit/train model
history = VGG_16.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)