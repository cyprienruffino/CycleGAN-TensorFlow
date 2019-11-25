import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl

import layers as cl


def create_network(inp, channels_out, name):
    with tf.name_scope(name):
        inp_lay = kl.Input(tensor=inp)

        # Encoder
        layer = kl.Conv2D(64, 7, padding="same", strides=1)(inp_lay)
        layer = cl.InstanceNormalization()(layer)
        layer = kl.Activation("relu")(layer)

        layer = kl.Conv2D(128, 3, padding="same", strides=2)(layer)
        layer = cl.InstanceNormalization()(layer)
        layer = kl.Activation("relu")(layer)

        layer = kl.Conv2D(256, 3, padding="same", strides=2)(layer)
        layer = cl.InstanceNormalization()(layer)
        layer = kl.Activation("relu")(layer)

        # Transformer
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)
        layer = cl.ResidualBlock(256, nb_layers=2, kernel_size=3, normalization="instancenorm")(layer)

        # Decoder
        layer = kl.Conv2DTranspose(128, 3, padding="same", strides=2)(layer)
        layer = cl.InstanceNormalization()(layer)
        layer = kl.Activation("relu")(layer)

        layer = kl.Conv2DTranspose(64, 3, padding="same", strides=2)(layer)
        layer = cl.InstanceNormalization()(layer)
        layer = kl.Activation("relu")(layer)

        G_out = kl.Conv2D(channels_out, 7, strides=1, activation="tanh", padding="same")(layer)

        model = k.Model(inputs=inp_lay, outputs=G_out)
    return model
