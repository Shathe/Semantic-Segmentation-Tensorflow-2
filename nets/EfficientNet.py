import tensorflow as tf
from tensorflow.keras import layers, regularizers


def reshape_into(inputs, input_to_copy):
    return tf.image.resize(inputs, (input_to_copy.shape[1], input_to_copy.shape[2]), method=tf.image.ResizeMethod.BILINEAR)


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
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0001),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0001),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


class EfficientConv(tf.keras.Model):
    def __init__(self, filters_in, filters_out, kernel_size=3, strides=1, expand_ratio=1, drop_rate=0.2, se_ratio=0., id_skip=True, dilation_rate=1):
        super(EfficientConv, self).__init__()

        self.filters_out = filters_out
        self.filters_in = filters_in
        self.expand_ratio = expand_ratio
        self.drop_rate = drop_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.id_skip = id_skip

        self.filters = filters_in * expand_ratio
        if expand_ratio != 1:
            self.conv_expand = convolution(self.filters, kernel_size=1, strides=1, dilation_rate=1, use_bias=False)
            self.bn_expand = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

        self.depthw = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=dilation_rate, use_bias=False)
        self.bn_depthw = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            self.global_avg = layers.GlobalAveragePooling2D()
            self.se_conv_1 = convolution(filters_se, kernel_size=1, strides=1, dilation_rate=1)
            self.se_conv_2 = convolution(self.filters, kernel_size=1, strides=1, dilation_rate=1)

        self.conv_out = convolution(self.filters_out, kernel_size=1, strides=1, dilation_rate=1, use_bias=False)
        self.bn_out = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

        self.drop = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))

    def call(self, inputs, activation=True, training=True):

        if self.expand_ratio != 1:
            x = self.conv_expand(inputs)
            x = self.bn_expand(x, training=training)
            x = tf.nn.swish(x)
        else:
            x = inputs

        x = self.depthw(x)
        x = self.bn_depthw(x, training=training)
        x = tf.nn.swish(x)

        if 0 < self.se_ratio <= 1:
            se = self.global_avg(x)
            se = layers.Reshape((1, 1, self.filters))(se)
            se = self.se_conv_1(se)
            se = tf.nn.swish(se)
            se = self.se_conv_2(se)
            se = tf.nn.swish(se)
            x = layers.multiply([x, se])

        x = self.conv_out(x)
        x = self.bn_out(x, training=training)

        if (self.id_skip is True and self.strides == 1 and self.filters_in == self.filters_out):
            if self.drop_rate > 0:
                x = self.drop(x, training=training)

            x = layers.add([x, inputs])

        # if activation:
        #     x = tf.nn.swish(x)

        return x


class EfficientConvDil(tf.keras.Model):
    def __init__(self, filters_in, filters_out, kernel_size=3, strides=1, expand_ratio=1, drop_rate=0.2, se_ratio=0., id_skip=True, dilation_rate=1):
        super(EfficientConvDil, self).__init__()

        self.filters_out = filters_out
        self.filters_in = filters_in
        self.expand_ratio = expand_ratio
        self.drop_rate = drop_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.id_skip = id_skip

        self.filters = filters_in * expand_ratio
        if expand_ratio != 1:
            self.conv_expand = convolution(self.filters, kernel_size=1, strides=1, dilation_rate=1, use_bias=False)
            self.bn_expand = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

        self.depthw = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=1, use_bias=False)
        self.bn_depthw = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)


        self.depthw_dil = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=dilation_rate, use_bias=False)
        self.bn_depthw_dil = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            self.global_avg = layers.GlobalAveragePooling2D()
            self.se_conv_1 = convolution(filters_se, kernel_size=1, strides=1, dilation_rate=1)
            self.se_conv_2 = convolution(self.filters, kernel_size=1, strides=1, dilation_rate=1)

        self.conv_out = convolution(self.filters_out, kernel_size=1, strides=1, dilation_rate=1, use_bias=False)
        self.bn_out = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

        self.drop = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))

    def call(self, inputs, activation=True, training=True):

        if self.expand_ratio != 1:
            x = self.conv_expand(inputs)
            x = self.bn_expand(x, training=training)
            x = tf.nn.swish(x)
        else:
            x = inputs

        x1 = self.depthw(x)
        x1 = self.bn_depthw(x1, training=training)
        x1 = tf.nn.swish(x1)

        x2 = self.depthw(x)
        x2 = self.bn_depthw(x2, training=training)
        x2 = tf.nn.swish(x2)

        x = x2 + x1

        if 0 < self.se_ratio <= 1:
            se = self.global_avg(x)
            se = layers.Reshape((1, 1, self.filters))(se)
            se = self.se_conv_1(se)
            se = tf.nn.swish(se)
            se = self.se_conv_2(se)
            se = tf.nn.swish(se)
            x = layers.multiply([x, se])

        x = self.conv_out(x)
        x = self.bn_out(x, training=training)

        if (self.id_skip is True and self.strides == 1 and self.filters_in == self.filters_out):
            if self.drop_rate > 0:
                x = self.drop(x, training=training)

            x = layers.add([x, inputs])

        if activation:
            x = tf.nn.swish(x)

        return x

class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x


class EfficientNetDil(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(EfficientNetDil, self).__init__(**kwargs)
        f = 24

        self.conv1 = Conv_BN(filters=f, kernel_size=3, strides=2)
        self.block1 = EfficientConv(filters_in=f, filters_out=f, kernel_size=3, strides=1, expand_ratio=1, drop_rate=0, se_ratio=0., id_skip=False)
        self.block2 = EfficientConv(filters_in=f, filters_out=f*2, kernel_size=3, strides=2, expand_ratio=1, drop_rate=0, se_ratio=0., id_skip=False)
        self.block3 = EfficientConv(filters_in=f*2, filters_out=f*2, kernel_size=3, strides=1, expand_ratio=3, drop_rate=0.1, se_ratio=0., id_skip=True)
        self.block6 = EfficientConv(filters_in=f*2, filters_out=f*4, kernel_size=3, strides=2, expand_ratio=3, drop_rate=0.1, se_ratio=0., id_skip=False)
        self.block7 = EfficientConvDil(filters_in=f*4, filters_out=f*4, kernel_size=3, strides=1, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=2)
        self.block8 = EfficientConvDil(filters_in=f*4, filters_out=f*4, kernel_size=3, strides=1, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=4)
        self.block9 = EfficientConvDil(filters_in=f*4, filters_out=f*4, kernel_size=3, strides=1, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=8)
        self.block94 = EfficientConvDil(filters_in=f*4, filters_out=f*4, kernel_size=5, strides=1, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=2)
        self.block10 = EfficientConvDil(filters_in=f*4, filters_out=f*4, kernel_size=5, strides=1, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=4)
        self.block11 = EfficientConvDil(filters_in=f*4, filters_out=f*8, kernel_size=5, strides=2, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=False, dilation_rate=8)

        self.block12 = EfficientConvDil(filters_in=f*8, filters_out=f*8, kernel_size=3, strides=1, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=2)
        self.block13 = EfficientConvDil(filters_in=f*8, filters_out=f*8, kernel_size=5, strides=1, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=4)
        self.block14 = EfficientConv(filters_in=f*8, filters_out=f*8, kernel_size=5, strides=1, expand_ratio=6, drop_rate=0.2, se_ratio=0.25, id_skip=True)

        self.block15_0 = EfficientConv(filters_in=f*8, filters_out=f*4, kernel_size=3, strides=1, expand_ratio=3, drop_rate=0., se_ratio=0., id_skip=False)
        self.block15 = EfficientConv(filters_in=f*4, filters_out=f*2, kernel_size=3, strides=1, expand_ratio=1, drop_rate=0., se_ratio=0., id_skip=False)
        self.block16 = EfficientConv(filters_in=f*2, filters_out=f*2, kernel_size=3, strides=1, expand_ratio=1, drop_rate=0., se_ratio=0., id_skip=False)

        self.classify = convolution(num_classes, 1, strides=1, dilation_rate=1, use_bias=True)


    def call(self, inputs, training=True):
        x1 = self.conv1(inputs, training=training)
        x = self.block1(x1, training=training)
        x2 = self.block2(x, training=training)
        x = self.block3(x2, training=training)
        x = self.block6(x, training=training)
        x3 = self.block7(x, training=training)
        x = self.block8(x3, training=training)
        x = self.block9(x, training=training)
        x = self.block94(x, training=training)
        x = self.block10(x, training=training)
        x4 = self.block11(x, training=training)
        x = self.block12(x4, training=training)
        x = self.block13(x, training=training)
        x = self.block14(x, training=training)
        x = reshape_into(x, x3)
        x = self.block15_0(x, training=training) + x3
        x = reshape_into(x, x2)
        x = self.block15(x, training=training) + x2
        x = reshape_into(x, x1)
        x = self.block16(x, training=training)
        x = reshape_into(x, inputs)
        x = self.classify(x, training=training)
        x = tf.keras.activations.softmax(x, axis=-1)
        return x


class EfficientNet(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)

        self.conv1 = Conv_BN(filters=32, kernel_size=3, strides=2)
        f = 32
        self.block1 = EfficientConv(filters_in=32, filters_out=f, kernel_size=3, strides=1, expand_ratio=1, drop_rate=0,
                                    se_ratio=0., id_skip=False)
        self.block2 = EfficientConv(filters_in=f, filters_out=f * 2, kernel_size=3, strides=2, expand_ratio=1,
                                    drop_rate=0, se_ratio=0., id_skip=False)
        self.block3 = EfficientConv(filters_in=f * 2, filters_out=f * 2, kernel_size=3, strides=1, expand_ratio=3,
                                    drop_rate=0.1, se_ratio=0., id_skip=True)
        self.block4 = EfficientConv(filters_in=f * 2, filters_out=f * 2, kernel_size=3, strides=1, expand_ratio=3,
                                    drop_rate=0.1, se_ratio=0., id_skip=True)
        self.block6 = EfficientConv(filters_in=f * 2, filters_out=f * 4, kernel_size=3, strides=2, expand_ratio=3,
                                    drop_rate=0.1, se_ratio=0., id_skip=False)
        self.block7 = EfficientConv(filters_in=f * 4, filters_out=f * 4, kernel_size=3, strides=1, expand_ratio=6,
                                    drop_rate=0.2, se_ratio=0.25, id_skip=True)
        self.block8 = EfficientConv(filters_in=f * 4, filters_out=f * 4, kernel_size=3, strides=1, expand_ratio=6,
                                    drop_rate=0.2, se_ratio=0.25, id_skip=True)
        self.block9 = EfficientConv(filters_in=f * 4, filters_out=f * 4, kernel_size=3, strides=1, expand_ratio=6,
                                    drop_rate=0.2, se_ratio=0.25, id_skip=True)
        self.block91 = EfficientConv(filters_in=f * 4, filters_out=f * 4, kernel_size=5, strides=1, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=True)
        self.block92 = EfficientConv(filters_in=f * 4, filters_out=f * 4, kernel_size=5, strides=1, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=True)
        self.block93 = EfficientConv(filters_in=f * 4, filters_out=f * 4, kernel_size=5, strides=1, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=True)
        self.block94 = EfficientConv(filters_in=f * 4, filters_out=f * 4, kernel_size=5, strides=1, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=True)
        self.block10 = EfficientConv(filters_in=f * 4, filters_out=f * 4, kernel_size=5, strides=1, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=True)
        self.block11 = EfficientConv(filters_in=f * 4, filters_out=f * 8, kernel_size=5, strides=2, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=False)

        self.block12 = EfficientConv(filters_in=f * 8, filters_out=f * 8, kernel_size=3, strides=1, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=2)
        self.block13 = EfficientConv(filters_in=f * 8, filters_out=f * 8, kernel_size=5, strides=1, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=2)
        self.block121 = EfficientConv(filters_in=f * 8, filters_out=f * 8, kernel_size=3, strides=1, expand_ratio=6,
                                      drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=2)
        self.block131 = EfficientConv(filters_in=f * 8, filters_out=f * 8, kernel_size=5, strides=1, expand_ratio=6,
                                      drop_rate=0.2, se_ratio=0.25, id_skip=True, dilation_rate=2)
        self.block14 = EfficientConv(filters_in=f * 8, filters_out=f * 8, kernel_size=5, strides=1, expand_ratio=6,
                                     drop_rate=0.2, se_ratio=0.25, id_skip=True)

        self.block15_0 = EfficientConv(filters_in=f * 8, filters_out=f * 4, kernel_size=3, strides=1, expand_ratio=3,
                                       drop_rate=0., se_ratio=0., id_skip=False)
        self.block15 = EfficientConv(filters_in=f * 4, filters_out=f * 2, kernel_size=3, strides=1, expand_ratio=1,
                                     drop_rate=0., se_ratio=0., id_skip=False)
        self.block16 = EfficientConv(filters_in=f * 2, filters_out=f * 2, kernel_size=3, strides=1, expand_ratio=1,
                                     drop_rate=0., se_ratio=0., id_skip=True)

        self.classify = convolution(num_classes, 1, strides=1, dilation_rate=1, use_bias=True)

    def call(self, inputs, training=True):
        x1 = self.conv1(inputs, training=training)
        x = self.block1(x1, training=training)
        x2 = self.block2(x, training=training)
        x = self.block3(x2, training=training)
        x = self.block4(x, training=training)
        x = self.block6(x, training=training)
        x3 = self.block7(x, training=training)
        x = self.block8(x3, training=training)
        x = self.block9(x, training=training)
        x = self.block91(x, training=training)
        x = self.block92(x, training=training)
        x = self.block93(x, training=training)
        x = self.block94(x, training=training)
        x = self.block10(x, training=training)
        x4 = self.block11(x, training=training)
        x = self.block12(x4, training=training)
        x = self.block13(x, training=training)
        x = self.block121(x, training=training)
        x = self.block131(x, training=training)
        x = self.block14(x, training=training)
        x = reshape_into(x, x3)
        x = self.block15_0(x, training=training) + x3
        x = reshape_into(x, x2)
        x = self.block15(x, training=training) + x2
        x = reshape_into(x, x1)
        x = self.block16(x, training=training)
        x = self.classify(x, training=training)
        x = reshape_into(x, inputs)
        x = tf.keras.activations.softmax(x, axis=-1)
        return x


