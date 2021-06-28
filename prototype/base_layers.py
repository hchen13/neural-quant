import tensorflow as tf
from tensorflow.python.keras.layers import Conv1D, BatchNormalization, ReLU


class ConvBn(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int=1, padding='same', act: bool=True, **kwargs):
        super(ConvBn, self).__init__(**kwargs)
        self.conv_params = dict(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )
        self.act = act

    def build(self, input_shape):
        self.conv_layer = Conv1D(**self.conv_params)
        self.bn_layer = BatchNormalization(
            scale=False,
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform'
        )
        if self.act:
            self.relu_layer = ReLU()

    def call(self, input_tensor, **kwargs):
        x = self.conv_layer(input_tensor)
        x = self.bn_layer(x)
        if self.act:
            x = self.relu_layer(x)
        return x