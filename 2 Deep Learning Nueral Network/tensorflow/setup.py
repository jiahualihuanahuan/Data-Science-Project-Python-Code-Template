# Using tensorflow backend
import keras

#-------------------------------------
# Using plaidml.keras.backend backend
import os, shutil
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras