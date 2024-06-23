# https://github.com/JierunChen/FasterNet

import tensorflow as tf

def to_3d(x):
    # Rearrange (batch_size, height*width, channels) to (batch_size, height, width, channels)
    return tf.reshape(x, [-1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]])

def to_4d(x, h, w):
    # Rearrange (batch_size, height, width, channels) to (batch_size, height*width, channels)
    return tf.reshape(x, [-1, h * w, tf.shape(x)[-1]])

class PartialConv3(tf.keras.layers.Layer):
    def __init__(self, dim, n_div, forward):
        super(PartialConv3, self).__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = tf.keras.layers.Conv2D(self.dim_conv3, 3, padding='same', use_bias=False)

        if forward == 'slicing':
            self.call = self.forward_slicing
        elif forward == 'split_cat':
            self.call = self.forward_split_cat
        else:
            raise NotImplementedError

    def call(self, x):
        if self.forward == self.forward_slicing:
            return self.forward_slicing(x)
        elif self.forward == self.forward_split_cat:
            return self.forward_split_cat(x)

    def forward_slicing(self, x):
        # only for inference
        x = tf.identity(x)  # Keep the original input intact for the residual connection later
        x_conv3 = self.partial_conv3(x[:, :, :, :self.dim_conv3])
        x = tf.concat([x_conv3, x[:, :, :, self.dim_conv3:]], axis=-1)

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = tf.split(x, [self.dim_conv3, self.dim_untouched], axis=-1)
        x1 = self.partial_conv3(x1)
        x = tf.concat([x1, x2], axis=-1)

        return x


# FasterBlock
class FasterNetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, n_div, expand_ratio=2, activation='relu', drop_path_rate=0.0, forward='split_cat'):
        super(FasterNetBlock, self).__init__()
        self.pconv = PartialConv3(dim, n_div, forward=forward)
        self.conv1 = tf.keras.layers.Conv2D(dim * expand_ratio, 1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act_layer = tf.keras.layers.Activation(activation)
        self.conv2 = tf.keras.layers.Conv2D(dim, 1, use_bias=False)
        self.drop_path = tf.keras.layers.Dropout(drop_path_rate) if drop_path_rate > 0.0 else tf.identity

    def call(self, x):
        height_width = int(x.shape[1] ** 0.5)  # Calculate height and width
        x = tf.reshape(x, [-1, height_width, height_width, x.shape[-1]])
        residual = x
        x = self.pconv(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        x = tf.reshape(x, [-1, height_width * height_width, x.shape[-1]])
        return x

