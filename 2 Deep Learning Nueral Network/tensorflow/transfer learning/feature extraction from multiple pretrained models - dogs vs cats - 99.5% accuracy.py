import tensorflow as tf
print(tf.__version__)
import h5py

train_data_directory = 'C:/Users/Jenny/python/Data/dog_vs_cat/train'
test_data_directory = 'C:/Users/Jenny/python/Data/dog_vs_cat/test'

def write_gap(MODEL, image_size, preprocess):
    width = image_size[0]
    height = image_size[1]
    input_tensor = tf.keras.Input((height, width, 3))
    x = input_tensor
    
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = tf.keras.Model(base_model.input, tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

    gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess)
    train_generator = gen.flow_from_directory(train_data_directory, image_size, shuffle=False, 
                                              batch_size=batch_size, class_mode='binary')
    test_generator = gen.flow_from_directory(test_data_directory, image_size, shuffle=False, 
                                             batch_size=batch_size, class_mode='binary')

    print('calculating training set')
    train = model.predict(train_generator, len(train_generator))
    print('calculating test set')
    test = model.predict(test_generator, len(test_generator))
    print('write file to disk')
    with h5py.File(f"C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_{model_name}.h5", 'w') as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


# VGG16
image_size = (224,224)
batch_size = 8
write_gap(tf.keras.applications.VGG16, 
		  image_size, 
		  tf.keras.applications.vgg16.preprocess_input)

# EfficientNetB7
image_size = (224,224)
batch_size = 8
write_gap(tf.keras.applications.EfficientNetB7, 
		  image_size, 
		  tf.keras.applications.efficientnet.preprocess_input)

# MobileNetV2
image_size = (224,224)
batch_size = 8
model_name = 'MobileNetV2'
write_gap(tf.keras.applications.MobileNetV2, 
		  image_size, 
		  tf.keras.applications.mobilenet_v2.preprocess_input)

# InceptionResNetV2
image_size = (299,299)
batch_size = 8
model_name = 'InceptionResNetV2'
write_gap(tf.keras.applications.InceptionResNetV2, 
		  image_size, 
		  tf.keras.applications.inception_resnet_v2.preprocess_input)

# DenseNet201
image_size = (224,224)
batch_size = 8
model_name = 'DenseNet201'
write_gap(tf.keras.applications.DenseNet201, 
		  image_size, 
		  tf.keras.applications.densenet.preprocess_input)

# NASNetLarge 
image_size = (331,331)
batch_size = 8
model_name = 'NASNetLarge'
write_gap(tf.keras.applications.NASNetLarge, 
		  image_size, 
		  tf.keras.applications.nasnet.preprocess_input)

# Xception
image_size = (299,299)
batch_size = 64
model_name = 'Xception'
write_gap(tf.keras.applications.Xception, 
		  image_size, 
		  tf.keras.applications.xception.preprocess_input)

# InceptionV3
image_size = (299,299)
batch_size = 4
model_name = 'InceptionV3'
write_gap(tf.keras.applications.InceptionV3, 
		  image_size, 
		  tf.keras.applications.inception_v3.preprocess_input)

#ResNet50
image_size = (224,224)
batch_size = 8
model_name = 'ResNet50'
write_gap(tf.keras.applications.ResNet50, 
		  image_size, 
		  tf.keras.applications.resnet50.preprocess_input)


import numpy as np
from sklearn.utils import shuffle
np.random.seed(2021)

X_train = []
X_test = []

for filename in ["C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_ResNet50.h5", 
                 "C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_Xception.h5", 
                 "C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_InceptionV3.h5", 
                 "C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_VGG16.h5", 
                 "C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_NASNetLarge.h5",
                "C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_EfficientNetB7.h5", 
                "C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_MobileNetV2.h5", 
                "C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_InceptionResNetV2.h5", 
                "C:/Users/Jenny/python/dogs_vs_cats/tensorflow/features/gap_DenseNet201.h5", ]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

# shuffle training set to avoid having only one class in validation set (see details in validation_split)

X_train, y_train = shuffle(X_train, y_train)

X_train.shape

X_test.shape

input_tensor = tf.keras.Input(X_train.shape[1:])
x = tf.keras.layers.Dropout(0.99)(input_tensor)
x = tf.keras.layers.Dense(1, activation='sigmoid')(input_tensor)
model = tf.keras.Model(input_tensor, x)

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint("C:/Users/Jenny/python/dogs_vs_cats/tensorflow/models/dog_vs_cat_feature_extraction.h5", 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto', 
                             save_freq='epoch')
early = tf.keras.callbacks.EarlyStopping(monitor='val_acc', 
                      min_delta=0, 
                      patience=10, 
                      verbose=1, 
                      mode='auto')
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.1,
                                            min_lr=0.000000001)

nb_epochs = 1000

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs = nb_epochs,
    validation_split=0.33,
    callbacks=[checkpoint, early, learning_rate_reduction])













































































