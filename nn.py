import tensorflow as tf


def loss(y, pred):
    return tf.keras.losses.binary_crossentropy(y, pred)


def accuracy(y, pred):
    return tf.keras.metrics.binary_accuracy(y, pred)
