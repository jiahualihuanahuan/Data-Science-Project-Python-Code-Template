import tensorflow as tf
print(tf.__version__)

# is_gpu_available will return true or false
print(tf.test.is_gpu_available())

# list_physical_devices will list all gpu devices
print(tf.config.list_physical_devices('GPU'))


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()