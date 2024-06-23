import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, initializers
import numpy as np
from Partial_conv3 import FasterNetBlock

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(
            units=projection_dim
        )
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(
            start=0,
            limit=self.num_patches,
            delta=1
        )
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        }) 
        return config

class MobileViTv2Attention(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__(**kwargs)
        self.d_model = d_model
        self.fc_i = tf.keras.layers.Dense(1)
        self.fc_k = tf.keras.layers.Dense(d_model)
        self.fc_v = tf.keras.layers.Dense(d_model)
        self.fc_o = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        i = self.fc_i(inputs)  # (bs,nq,1)
        weight_i = tf.nn.softmax(i, axis=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(inputs)  # bs,nq,d_model
        context_vector = tf.reduce_sum(context_score, axis=1, keepdims=True)  # bs,1,d_model
        v = self.fc_v(inputs) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model
        return out

    def get_config(self):
        config = super(MobileViTv2Attention, self).get_config()
        config.update({'d_model': self.d_model})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = tf.exp(-gamma * labels * logits - gamma * tf.math.log(1 +
            tf.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = tf.reduce_sum(weighted_loss)

    focal_loss /= tf.reduce_sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, classes, loss_type, beta, gamma):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * classes

    # labels_one_hot = tf.one_hot(labels, depth=classes)
    labels_one_hot = labels

    weights = tf.constant(weights, dtype=tf.float32)
    weights = tf.expand_dims(weights, axis=0)
    weights = tf.tile(weights, [tf.shape(labels_one_hot)[0], 1]) * labels_one_hot
    weights = tf.reduce_sum(weights, axis=1)
    weights = tf.expand_dims(weights, axis=1)
    weights = tf.tile(weights, [1, classes])

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels_one_hot, pos_weight=weights)
    elif loss_type == "softmax":
        pred = tf.nn.softmax(logits)
        cb_loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels_one_hot, pos_weight=weights)
    return cb_loss

def create_vit_classifier(input_shape, patch_size, num_patches, projection_dim, transformer_layers,
                          num_heads, transformer_units, mlp_head_units, num_classes):
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs) 
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches) # encoded_patches.shape = (None,36,36)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.5
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def create_model_Fastertransformerblock(input_shape, patch_size, num_patches, projection_dim, transformer_layers,
                                       transformer_units, mlp_head_units, num_classes):
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        fasterblock_output = FasterNetBlock(dim=projection_dim, n_div=4)(encoded_patches)
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MobileViTv2Attention(d_model=projection_dim)(x1)
        # Skip connection 1. 
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.5)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

        encoded_patches = layers.Add()([encoded_patches, fasterblock_output])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# Ablation settings
def create_ablation(input_shape, patch_size, num_patches, projection_dim, transformer_layers,
                    num_heads, transformer_units, mlp_head_units, num_classes):
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        # fasterblock_output = FasterNetBlock(dim=projection_dim, n_div=4)(encoded_patches)
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MobileViTv2Attention(d_model=projection_dim)(x1)
        # attention_output = layers.MultiHeadAttention(
        #     num_heads=num_heads, key_dim=projection_dim, dropout=0.5
        # )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.5)  # 改成0.5
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

        # encoded_patches = layers.Add()([encoded_patches, fasterblock_output])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features) 
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
