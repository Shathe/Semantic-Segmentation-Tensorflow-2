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

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
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

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x



class Residual_SeparableConv(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(Residual_SeparableConv, self).__init__()

        self.conv = DepthwiseConv_BN(filters, kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.dropout = layers.Dropout(rate=dropout)
    def call(self, inputs, training=True):

        x = self.conv(inputs, activation=False, training=training)
        x = self.dropout(x, training=training)
        if inputs.shape == x.shape:
            x = x + inputs
        x = layers.ReLU()(x)

        return x

class Residual_SeparableConv_dil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(Residual_SeparableConv_dil, self).__init__()
        self.conv1 = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=1, use_bias=False)
        self.conv2 = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=dilation_rate, use_bias=False)
        self.bn1 = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)
        self.bn2 = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)


        self.conv = convolution(filters=filters, kernel_size=1, strides=1, dilation_rate=1)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

        self.dropout = layers.Dropout(rate=dropout)
    def call(self, inputs, training=True):

        x1 = self.conv1(inputs)
        x1 = self.bn1(x1, training=training)
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2, training=training)
        x1 = layers.ReLU()(x1)
        x2 = layers.ReLU()(x2)

        x = self.conv(x1 + x2)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)

        if inputs.shape == x.shape:
            x = x + inputs

        x = layers.ReLU()(x)

        return x


class MininetV2Module(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(MininetV2Module, self).__init__()

        self.conv1 = Residual_SeparableConv(filters, kernel_size, strides=strides, dilation_rate=1, dropout=dropout)
        self.conv2 = Residual_SeparableConv(filters, kernel_size, strides=1, dilation_rate=dilation_rate, dropout=dropout)


    def call(self, inputs, training=True):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x


class MininetV2Module_dil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(MininetV2Module_dil, self).__init__()

        self.conv1 = Residual_SeparableConv_dil(filters, kernel_size, strides=strides, dilation_rate=1, dropout=dropout)
        self.conv2 = Residual_SeparableConv_dil(filters, kernel_size, strides=1, dilation_rate=dilation_rate, dropout=dropout)


    def call(self, inputs, training=True):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x


class MininetV2Downsample(tf.keras.Model):
    def __init__(self, filters, depthwise=True):
        super(MininetV2Downsample, self).__init__()
        if depthwise:
            self.conv = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)
        else:
            self.conv = Conv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)

    def call(self, inputs, training=True):

        x = self.conv(inputs, training=training)

        return x


class MininetV2Upsample(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(MininetV2Upsample, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, last=False, training=True):
        x = self.conv(inputs)
        if not last:
            x = self.bn(x, training=training)
            x = layers.ReLU()(x)

        return x





class ShatheBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size,  dilation_rate=1):
        super(ShatheBlock, self).__init__()

        self.kernel_size = kernel_size
        self.filters = filters

        self.conv1 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv2 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv3 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)

    def call(self, inputs, training=True):
        x2 = self.conv1(inputs, training=training)
        x3 = self.conv2(x2, training=training)
        x = self.conv3(x3, activation=False, training=training)
        if inputs.shape[3] == x.shape[3]:
            return layers.ReLU()(x + inputs)
        else:
            return layers.ReLU()(x2 + x)




class ResNet50Seg(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(ResNet50Seg, self).__init__(**kwargs)
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights=weights,
                                                             input_shape=input_shape, pooling='avg')

        output_2 = base_model.get_layer('conv2_block2_out').output
        output_3 = base_model.get_layer('conv3_block3_out').output
        output_4 = base_model.get_layer('conv4_block5_out').output
        output_5 = base_model.get_layer('conv5_block3_out').output
        outputs = [output_5, output_4, output_3, output_2]

        self.model_output = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        self.conv_up1 = Conv_BN(1024, 3)
        self.conv_up2 = Conv_BN(512, 3)
        self.conv_up3 = Conv_BN(256, 3)

        self.classify = convolution(num_classes, 1, strides=1, dilation_rate=1, use_bias=True)

    def call(self, inputs, training=True):

        outputs = self.model_output(inputs, training=training)

        x = reshape_into(outputs[0], outputs[1])
        x = self.conv_up1(x, training=training) + outputs[1]
        x = reshape_into(x, outputs[2])


        x = self.conv_up2(x, training=training) + outputs[2]
        x = reshape_into(x, outputs[3])


        x = self.conv_up3(x, training=training) + outputs[3]
        x = self.classify(x, training=training)

        x = reshape_into(x, inputs)

        x = tf.keras.activations.softmax(x, axis=-1)

        return x


 
