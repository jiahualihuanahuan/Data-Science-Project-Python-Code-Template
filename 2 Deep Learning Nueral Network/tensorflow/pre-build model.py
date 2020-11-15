# import/load pre-build model VGG16
from keras.applications.vgg16 import VGG16
model = VGG16()
print(model.summary())

# create a plot of VGG16 graph
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
model = VGG16()
plot_model(model, to_file='vgg.png')

# make prediction using the pre-build model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


for i in range (1,12500):
    # load an image from file
    image = load_img(f'E:/Dropbox/Machine Learning/Data/dogs-vs-cats/{i}.jpg', target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print(f'picture {i}')
    print('%s (%.2f%%)' % (label[1], label[2]*100))