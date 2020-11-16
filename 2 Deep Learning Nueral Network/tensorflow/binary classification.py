# build a VGG16 neural network model
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

VGG_16 = Sequential()
VGG_16.add(Conv2D(input_shape=(224, 224, 3), filters=64,
                  kernel_size=(3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=64, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Flatten())
VGG_16.add(Dense(4096, activation="relu"))
VGG_16.add(Dense(4096, activation="relu"))
VGG_16.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

VGG_16.summary()

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

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
#----------------------------------------------
# change 0/1 to cat and dog
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

# build a VGG16 neural network model
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

VGG_16 = Sequential()
VGG_16.add(Conv2D(input_shape=(224, 224, 3), filters=64,
                  kernel_size=(3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=64, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
VGG_16.add(Flatten())
VGG_16.add(Dense(4096, activation="relu"))
VGG_16.add(Dense(4096, activation="relu"))
# 2 neron and softmax activation function here
VGG_16.add(Dense(2, activation="softmax"))

model.compile(loss='categorical_crossentropy',  # categorical_crossentropy here
              optimizer='rmsprop', metrics=['accuracy'])

VGG_16.summary()

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "../input/train/train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',  # categorical here
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "../input/train/train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',  # categorical here
    batch_size=batch_size
)
#----------------------------------------------
