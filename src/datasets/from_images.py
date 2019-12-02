import os
from functools import partial

import tensorflow as tf
import numpy as np


def _parse(filename, channels):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=channels)
    return tf.cast(image_decoded, tf.float32)


def _flip(x):
    x = tf.image.random_flip_left_right(x)
    return x


def _crop(x, crop_size):
    shape = x.shape
    topleft_x = tf.random.uniform((1,), minval=0, maxval=(shape[0] - crop_size), dtype=tf.int32)
    topleft_y = tf.random.uniform((1,), minval=0, maxval=(shape[1] - crop_size), dtype=tf.int32)
    return tf.image.crop_to_bounding_box(x, topleft_y[0], topleft_x[0], crop_size, crop_size)


def _resize(x, resize_size):
    return tf.image.resize(x, [resize_size, resize_size])


def iterator(path, dataset_size, batch_size, channels, resize_size, crop_size):
    filenames = list(map(lambda p: os.path.join(path, p), os.listdir(path)))[:dataset_size]
    files = tf.constant(filenames)

    dataset = tf.data.Dataset.from_tensor_slices(files) \
        .map(partial(_parse, channels=channels)) \
        .map(lambda x: (x / 127.5) - 1) \
        .map(_flip) \
        .map(partial(_resize, resize_size=resize_size))\
        .map(partial(_crop, crop_size=crop_size)) \
        .shuffle(buffer_size=500) \
        .batch(batch_size) \
        .repeat()

    return dataset.make_initializable_iterator()


def oneshot_iterator(path, dataset_size, batch_size, channels):
    filenames = list(map(lambda p: os.path.join(path, p), os.listdir(path)))[:dataset_size]
    files = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(files) \
        .map(partial(_parse, channels=channels)) \
        .map(lambda x: (x / 127.5) - 1) \
        .batch(batch_size)

    return dataset.make_one_shot_iterator()
