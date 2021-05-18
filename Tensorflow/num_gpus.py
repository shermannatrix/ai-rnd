import tensorflow as tf
print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))