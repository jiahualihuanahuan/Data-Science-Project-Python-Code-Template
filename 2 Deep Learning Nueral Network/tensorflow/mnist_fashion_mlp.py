# setup: import libraries and modules
import tensorflow as tf
from tensorflow import keras

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

# Define Sequential model with 3 layers
model = keras.Sequential(
    [	
    	keras.layers.Flatten(input_shape = (28,28)),
        keras.layers.Dense(128, activation="relu", name="layer1"),
        keras.layers.Dense(64, activation="relu", name="layer2"),
        keras.layers.Dense(32, activation="relu", name="layer3"),
        keras.layers.Dense(16, activation="relu", name="layer4"),
        keras.layers.Dense(10, activation="softmax", name="layer5"),
    ]
)

print(model.summary())

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['acc'])

model.fit(train_images, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test ACC: {test_acc}")