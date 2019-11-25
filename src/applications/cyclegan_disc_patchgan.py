import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl

from layers import InstanceNormalization


def create_network(inp, channels, name, patch_size=70):
    with tf.name_scope(name):
        inp_layer = kl.Input(tensor=inp)

        # Discriminator
        layer = kl.Lambda(lambda x: tf.random_crop(x, [-1, patch_size, patch_size, channels]))(inp_layer)
        layer = kl.Conv2D(64, 4, padding="same", strides=2)(layer)
        layer = InstanceNormalization()(layer)
        layer = kl.LeakyReLU(0.2)(layer)

        layer = kl.Conv2D(128, 4, padding="same", strides=2)(layer)
        layer = InstanceNormalization()(layer)
        layer = kl.LeakyReLU(0.2)(layer)

        layer = kl.Conv2D(256, 4, padding="same", strides=2)(layer)
        layer = InstanceNormalization()(layer)
        layer = kl.LeakyReLU(0.2)(layer)

        layer = kl.Conv2D(512, 4, padding="same", strides=2)(layer)
        layer = InstanceNormalization()(layer)
        layer = kl.LeakyReLU(0.2)(layer)

        # D_out = kl.Conv2D(1, 4, activation="sigmoid", padding="same")(layer)
        D_out = kl.Conv2D(1, 4, padding="same")(layer)

        model = k.Model(inputs=inp_layer, outputs=D_out)
    return model
