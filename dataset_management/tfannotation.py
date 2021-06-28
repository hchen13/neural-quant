import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(history, label, future):
    feature = {
        'history': _bytes_feature(tf.io.serialize_tensor(history)),
        'label': _int64_feature(label),
        'future': _bytes_feature(tf.io.serialize_tensor(future))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def read_tfrecord(serialized_example):
    feature_description = {
        'history': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'future': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    history = tf.io.parse_tensor(example['history'], out_type=tf.float32)
    label = example['label']
    future = tf.io.parse_tensor(example['future'], out_type=tf.float32)
    return history, label, future
