import tensorflow as tf


def loss(y, pred):
    pred = tf.nn.softmax(pred)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)

    mask = tf.cast(tf.math.not_equal(y, 0), dtype=loss.dtype)

    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
