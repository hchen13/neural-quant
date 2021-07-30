import numpy as np
import tensorflow as tf


@tf.function
def weighted_categorical_crossentropy(y_true, y_pred, weights: list=None):
    if weights is None:
        weights = [1., 1., 1.]
    assert len(weights) == 3
    weights = tf.constant(weights, dtype=y_true.dtype)
    epsilon_ = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon_, 1 - epsilon_)
    losses = -y_true * tf.math.log(y_pred) * weights
    losses = tf.reduce_sum(losses, axis=-1)
    return tf.reduce_mean(losses)
