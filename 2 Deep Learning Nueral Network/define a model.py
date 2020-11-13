# There are a few ways to define the Sequential model
# import required libraries and modules
import tensorflow as tf
from tensorflow import keras

# 1 use Sequential([...])
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', name='layer 1'),
    keras.layers.Dense(16, activation='relu', name='layer 2'),
    keras.layers.Dense(8, activation='relu', name='layer 3'),
])
print(model.summary())

# 2 use add(...)
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation='relu', name='layer 1'))
model.add(keras.layers.Dense(16, activation='relu', name='layer 2'))
model.add(keras.layers.Dense(8, activation='relu', name='layer 3'))
print(model.summary())
model.pop()
print(len(model.layers))
print(model.summary())
