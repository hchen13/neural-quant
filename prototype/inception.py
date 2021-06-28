import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization, ReLU, Concatenate, MaxPooling1D, Conv1D

from prototype.base_layers import ConvBn


class Inception(tf.keras.layers.Layer):
    def __init__(self, f_1: int, f_2: list, f_3: list, f_out: int, reduction: bool, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.params = dict(
            f_1=f_1,
            f_2=f_2,
            f_3=f_3,
            f_out=f_out,
            strides=2 if reduction else 1,
            reduction=reduction
        )

    def build(self, input_shape):
        self.bn_layer = BatchNormalization(scale=False)
        self.relu_layer = ReLU()
        self.concat_layer = Concatenate()

        self.conv1 = ConvBn(
            filters=self.params['f_1'],
            kernel_size=1,
            strides=self.params['strides'],
            name='incep1'
        )

        self.conv2 = [
            ConvBn(filters=self.params['f_2'][0], kernel_size=1, strides=self.params['strides'], name='incep2_1'),
            ConvBn(filters=self.params['f_2'][1], kernel_size=3, name='incep2_2')
        ]

        self.conv3 = [
            ConvBn(filters=self.params['f_3'][0], kernel_size=1, strides=self.params['strides'], name='incep3_1'),
            ConvBn(filters=self.params['f_3'][1], kernel_size=5, name='incep3_2')
        ]
        if self.params['reduction']:
            self.pool_layers = [
                MaxPooling1D(pool_size=3, strides=2, padding='same', name='incep4_pool'),
                ConvBn(filters=128, kernel_size=1, name='incep4_conv')
            ]
        self.conv_out = Conv1D(filters=self.params['f_out'], kernel_size=1, use_bias=False, name='out', padding='same')
        self.projection_layer = Conv1D(
            filters=self.params['f_out'], kernel_size=1, strides=2, use_bias=False, padding='same', name='projection')

    def call(self, input_tensor, **kwargs):
        layer_in = self.bn_layer(input_tensor)
        layer_in = self.relu_layer(layer_in)

        branch_outputs = [self.conv1(layer_in)]
        x = layer_in
        for layer in self.conv2:
            x = layer(x)
        branch_outputs.append(x)

        x = layer_in
        for layer in self.conv3:
            x = layer(x)
        branch_outputs.append(x)

        if self.params['reduction']:
            x = layer_in
            for layer in self.pool_layers:
                x = layer(x)
            branch_outputs.append(x)
            input_tensor = self.projection_layer(input_tensor)

        concat = self.concat_layer(branch_outputs)
        x = self.conv_out(concat)

        return x + input_tensor