import tensorflow as tf
from tensorflow.keras import layers, regularizers


def upsampling(inputs, scale):

    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale],
                                    align_corners=True)


def reshape_into(inputs, input_to_copy):
    return tf.image.resize_bilinear(inputs, [input_to_copy.get_shape()[1].value,
                                             input_to_copy.get_shape()[2].value], align_corners=True)


# convolution
def convolution(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Traspose convolution
def transposeConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=True):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0003),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0003),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


class DepthwiseConv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        # separableConv
        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, activation=True):
        x = self.conv(inputs)
        x = self.bn(x)
        if activation:
            x = layers.ReLU()(x)

        return x



class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, activation=True):
        x = self.conv(inputs)
        x = self.bn(x)
        if activation:
            x = layers.ReLU()(x)

        return x



class Residual_SeparableConv(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(Residual_SeparableConv, self).__init__()

        self.conv = DepthwiseConv_BN(filters, kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.dropout = layers.Dropout(rate=dropout)
    def call(self, inputs):

        x = self.conv(inputs, activation=False)
        x = self.dropout(x)
        x = layers.ReLU()(x + inputs)

        return x


class MininetV2Module(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(MininetV2Module, self).__init__()

        self.conv1 = Residual_SeparableConv(filters, kernel_size, strides=strides, dilation_rate=1, dropout=dropout)
        self.conv2 = Residual_SeparableConv(filters, kernel_size, strides=1, dilation_rate=dilation_rate, dropout=dropout)


    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)

        return x

class MininetV2Downsample(tf.keras.Model):
    def __init__(self, filters, depthwise=True):
        super(MininetV2Downsample, self).__init__()
        if depthwise:
            self.conv = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)
        else:
            self.conv = Conv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)

    def call(self, inputs):

        x = self.conv(inputs)
        return x


class MininetV2Upsample(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(MininetV2Upsample, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, last=False):
        x = self.conv(inputs)
        if not last:
            x = self.bn(x)
            x = layers.ReLU()(x)

        return x


class MiniNetv2(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(MiniNetv2, self).__init__(**kwargs)

        self.down1 = MininetV2Downsample(16, depthwise=False)
        self.down2 = MininetV2Downsample(64, depthwise=True)
        self.conv_mod_1 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_2 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_3 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_4 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.down3 = MininetV2Downsample(128, depthwise=True)
        self.conv_mod_5 = MininetV2Module(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_6 = MininetV2Module(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_7 = MininetV2Module(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_8 = MininetV2Module(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_9 = MininetV2Module(128, 3, strides=1, dilation_rate=16)
        self.conv_mod_10 = MininetV2Module(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_11 = MininetV2Module(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_12 = MininetV2Module(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_13 = MininetV2Module(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_14 = MininetV2Module(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_15 = MininetV2Module(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_16 = MininetV2Module(128, 3, strides=1, dilation_rate=4)
        self.upsample1 = MininetV2Upsample(64, 3, strides=2)
        self.conv_mod_17 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_18 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_19 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.upsample2 = MininetV2Upsample(num_classes, 3, strides=2)

    def call(self, inputs):
        x = self.down1(inputs)
        x = self.down2(x)
        x = self.conv_mod_1(x)
        x = self.conv_mod_2(x)
        x = self.conv_mod_3(x)
        x = self.conv_mod_4(x)
        x = self.down3(x)
        x = self.conv_mod_5(x)
        x = self.conv_mod_6(x)
        x = self.conv_mod_7(x)
        x = self.conv_mod_8(x)
        x = self.conv_mod_9(x)
        x = self.conv_mod_10(x)
        x = self.conv_mod_11(x)
        x = self.conv_mod_12(x)
        x = self.conv_mod_13(x)
        x = self.conv_mod_14(x)
        x = self.conv_mod_15(x)
        x = self.conv_mod_16(x)
        x = self.upsample1(x)
        x = self.conv_mod_17(x)
        x = self.conv_mod_18(x)
        x = self.conv_mod_19(x)
        x = self.upsample2(x, last=True)
        x = tf.keras.activations.softmax(x, axis=-1)

        return x
 