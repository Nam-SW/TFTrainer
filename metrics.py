import tensorflow as tf


def loss(y, pred):
    pred = tf.nn.softmax(pred)
    idx_except_last = tf.meshgrid(
        *[tf.range(s) for s in pred.shape[:-1]], indexing="ij"
    )
    idx = tf.stack(idx_except_last + [y], axis=-1)
    pred_ = tf.gather_nd(pred, idx)

    loss = tf.math.log(pred_)
    mask = tf.cast(tf.math.not_equal(y, 0), dtype=loss.dtype)

    loss = -tf.reduce_sum(loss * mask, axis=-1)
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
