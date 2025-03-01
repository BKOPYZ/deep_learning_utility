import tensorflow as tf

# Check if TensorFlow can access a GPU
if tf.config.list_physical_devices("GPU"):
    print("TensorFlow can use GPU")
else:
    print("TensorFlow cannot use GPU")
