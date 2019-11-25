import os
import sys

import cv2
import numpy as np
import progressbar
from tensorflow.keras.models import load_model
import tensorflow as tf


def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx


def tf_sqrtm_sym(mat, eps=1e-10):
    # WARNING : This only works for symmetric matrices !
    s, u, v = tf.svd(mat)
    si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    return tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)


def create_fid_func(model_path, sess, nb_data, lays_cut=1):
    with tf.name_scope("FID_activations"):
        cl = load_model(model_path)
        fid_model = tf.keras.Model(cl.inputs, cl.layers[-lays_cut - 1].output)

    with tf.name_scope("FID"):
        acts_real_pl = tf.placeholder(tf.float32, (nb_data, None))
        acts_fake_pl = tf.placeholder(tf.float32, (nb_data, None))

        mu_real = tf.reduce_mean(acts_real_pl, axis=0)
        mu_fake = tf.reduce_mean(acts_fake_pl, axis=0)

        sigma_real = tf_cov(acts_real_pl)
        sigma_fake = tf_cov(acts_fake_pl)
        diff = mu_real - mu_fake
        mu2 = tf.reduce_sum(tf.multiply(diff, diff))

        # Computing the sqrt of sigma_real * sigma_fake
        # See https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
        sqrt_sigma = tf_sqrtm_sym(sigma_real)
        # sqrt_sigma = tf.transpose(sigma_fake)
        sqrt_a_sigma_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_fake, sqrt_sigma))

        tr = tf.trace(sigma_real + sigma_fake - 2 * tf_sqrtm_sym(sqrt_a_sigma_a))
        fid = mu2 + tr

    def _fid(real, fake, batch_size=8):
        acts_real_val = []
        print("Computing activations on real data")
        for X in progressbar.ProgressBar()(range(0, len(real), batch_size)):
            if len(real[X:X + batch_size]) == batch_size:
                acts_real_val.append(np.reshape(fid_model.predict(real[X:X + batch_size]), (batch_size, -1)))

        acts_fake_val = []
        print("Computing activations on fake data")
        for X in progressbar.ProgressBar()(range(0, len(fake), batch_size)):
            if len(fake[X:X + batch_size]) == batch_size:
                acts_fake_val.append(np.reshape(fid_model.predict(fake[X:X + batch_size]), (batch_size, -1)))

        acts_fake_val = np.concatenate(acts_fake_val)
        acts_real_val = np.concatenate(acts_real_val)

        print("Computing FID")
        return sess.run(fid, feed_dict={acts_real_pl: acts_real_val, acts_fake_pl: acts_fake_val})

    return _fid


def eval_test(fidmodel, reals, fakes, batch_size=10):
    real_imgs = []
    fake_imgs = []
    for img in os.listdir(reals):
        real_imgs.append(cv2.imread(os.path.join(reals, img)))
    for img in os.listdir(fakes):
        fake_imgs.append(cv2.imread(os.path.join(fakes, img)))

    with tf.Session() as sess:
        fid_fun = create_fid_func(fidmodel, sess, len(real_imgs), lays_cut=4)
        fid = fid_fun(np.asarray(real_imgs), np.asarray(fake_imgs), batch_size)

    with open("output.txt", "w") as f:
        f.write(str(fid))
    print(fid)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        eval_test(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage : evaluate_fid.py fidmodel_path original_path generated_path")
