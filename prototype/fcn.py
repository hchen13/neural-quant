from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import MaxPooling1D, Dense, BatchNormalization, ReLU, GlobalAveragePooling1D

from prototype.base_layers import ConvBn
from prototype.inception import Inception


def build_fcn(input_size):
    input_tensor = Input(shape=(input_size, 5), name='input')

    conv11 = ConvBn(filters=8, kernel_size=7, name='conv1_1')(input_tensor)
    conv12 = ConvBn(filters=16, kernel_size=3, name='conv1_2')(conv11)
    pool11 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool1_1')(conv12)

    conv21 = ConvBn(filters=16, kernel_size=3, name='conv2_1')(pool11)
    conv22 = ConvBn(filters=32, kernel_size=3, name='conv2_2')(conv21)

    conv31 = Inception(f_1=32, f_2=[32, 64], f_3=[32, 64], f_out=64, reduction=True, name='conv3_1')(conv22)
    conv32 = Inception(f_1=32, f_2=[32, 64], f_3=[32, 64], f_out=64, reduction=False, name='conv3_2')(conv31)
    conv33 = Inception(f_1=32, f_2=[32, 64], f_3=[32, 64], f_out=64, reduction=False, name='conv3_3')(conv32)

    conv41 = Inception(f_1=32, f_2=[64, 96], f_3=[64, 64], f_out=128, reduction=True, name='conv4_1')(conv33)
    conv42 = Inception(f_1=32, f_2=[64, 96], f_3=[64, 64], f_out=128, reduction=False, name='conv4_2')(conv41)
    conv43 = Inception(f_1=32, f_2=[64, 96], f_3=[64, 64], f_out=128, reduction=False, name='conv4_3')(conv42)

    conv51 = Inception(f_1=64, f_2=[64, 192], f_3=[64, 192], f_out=256, reduction=True, name='conv5_1')(conv43)
    conv52 = Inception(f_1=64, f_2=[64, 192], f_3=[64, 192], f_out=256, reduction=False, name='conv5_2')(conv51)
    conv53 = Inception(f_1=64, f_2=[64, 192], f_3=[64, 192], f_out=256, reduction=False, name='conv5_3')(conv52)

    conv61 = Inception(f_1=128, f_2=[128, 256], f_3=[128, 256], f_out=512, reduction=True, name='conv6_1')(conv53)
    conv62 = Inception(f_1=128, f_2=[128, 256], f_3=[128, 256], f_out=512, reduction=False, name='conv6_2pre')(conv61)
    conv62 = BatchNormalization(
        scale=False,
        beta_initializer='glorot_uniform',
        gamma_initializer='glorot_uniform',
        name='conv6_2bn'
    )(conv62)
    conv62 = ReLU(name='conv6_2')(conv62)

    features = GlobalAveragePooling1D(name='features')(conv62)
    out = Dense(3, activation='softmax', name='prediction')(features)
    model = Model(input_tensor, out, name='FCN')
    return model


if __name__ == '__main__':
    fcn = build_fcn(144)
    fcn.summary()