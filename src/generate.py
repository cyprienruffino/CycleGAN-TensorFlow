import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import cv2
import numpy as np
import progressbar
import tensorflow as tf

from layers.instancenorm import InstanceNormalization


def main(checkpoint_path, files_path, output_path):
    mod = tf.keras.models.load_model(checkpoint_path, custom_objects={"InstanceNormalization": InstanceNormalization})
    inp = tf.keras.layers.Input((None, None, mod.input_shape[-1]))
    mod = tf.keras.models.Model(inp, mod(inp))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for name in progressbar.ProgressBar()(os.listdir(files_path)):
        img = (cv2.imread(os.path.join(files_path, name), cv2.IMREAD_UNCHANGED) / 127.5) - 1
        out = (mod.predict(np.expand_dims(img, axis=0)) + 1) * 127.5
        cv2.imwrite(os.path.join(output_path, name), out[0])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate.py checkpoint_path files_path output_path")
    main(sys.argv[1], sys.argv[2], sys.argv[3])