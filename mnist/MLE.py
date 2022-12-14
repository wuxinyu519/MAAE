"""
Deterministic supervised adversarial autoencoder.

 We are using:
    - Gaussian distribution as prior distribution.
    - dense layers.
    - Cyclic learning rate
"""
import tensorflow as tf
import gc
import os
import time
import numpy as np
from pathlib import Path

from mindspore import context
import mindspore.numpy
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt

PROJECT_ROOT = Path.cwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# -------------------------------------------------------------------------------------------------------------
# Set random seed
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# -------------------------------------------------------------------------------------------------------------
img_size = 28
num_c = 1
z_dim =8
BUFFER_SIZE = 60000
BATCH_SIZE = 100
sigma = None
squeeze1 = mindspore.ops.Squeeze(1)
# -------------------------------------------------------------------------------------------------------------
# Loading data
# -------------------------------------------------------------------------------------------------------------
print("Loading data...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator


x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train,  # 把上面剩余的 x_train, y_train继续拿来切
    test_size=1 / 60  # test_size默认是0.25
)
x_valid = x_valid.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_valid = x_valid.reshape(x_valid.shape[0], img_size * img_size * num_c)
x_test = x_test.reshape(x_test.shape[0], img_size * img_size * num_c)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
def gaussian_mixture(batch_size, labels, n_classes):
    x_stddev = 5.
    y_stddev = 1.
    shift = 10.

    x = np.random.normal(0, x_stddev, batch_size).astype("float32") + shift

    y = np.random.normal(0, y_stddev, batch_size).astype("float32")
    z = np.array([[xx, yy] for xx, yy in zip(x, y)])

    def rotate(z, label):
        angle = label * 2.0 * np.pi / n_classes
        rotation_matrix = np.array(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        )
        z[np.where(labels == label)] = np.array(
            [
                rotation_matrix.dot(np.array(point))
                for point in z[np.where(labels == label)]
            ]
        )
        return z

    for label in set(labels):
        rotate(z, label)

    return z


UNIQUE_RUN_ID = 'unsupervised_3d_maae_max_gaussian_posterior_dense_8z_test'
# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator
n_labels=10
# =============supervised=================
# label_sample = np.random.randint(0, n_labels, size=[x_test.shape[0]])
# z = gaussian_mixture(x_test.shape[0], label_sample, n_labels)
# =============unsupervised=================
z = tf.random.normal([x_test.shape[0], z_dim], mean=0.0, stddev=5.)

decoder = tf.keras.models.load_model(UNIQUE_RUN_ID+"/decoder_199.model/")
# generator = tf.keras.models.load_model("./gan_75z/generator_24.model/")


def log_mean_exp(a):
    max_ = a.max(axis=1)
    max2 = np.reshape(max_, (max_.shape[0], 1))
    return max_ + np.log(np.exp(a - max2).mean(1))


def mind_parzen(x, mu, sigma):
    a = (np.reshape(x, (x.shape[0], 1, x.shape[-1])) - np.reshape(mu, (1, mu.shape[0], mu.shape[-1]))) / sigma
    a5 = -0.5 * (a ** 2).sum(2)
    E = log_mean_exp(a5)
    t4 = sigma * np.sqrt(np.pi * 2)
    t5 = np.log(t4)
    Z = mu.shape[1] * t5
    return E - Z


def get_nll(x, samples, sigma, batch_size):
    '''get_nll'''
    inds = range(x.shape[0])
    inds = list(inds)
    n_batches = int(np.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = Tensor(np.array([]).astype(np.float32))
    for i in range(n_batches):
        begin = time.time()
        nll = mind_parzen(x[inds[i::n_batches]], samples, sigma)
        end = time.time()
        times.append(end - begin)
        nlls = tf.concat([nlls, nll], 0)

        if i % 10 == 0:
            print(i, np.mean(times), np.mean(nlls))

    return nlls


def cross_validate_sigma(samples, data, sigmas, batch_size):
    '''cross_validate_sigma'''
    lls = Tensor(np.array([]).astype(np.float32))
    for sigma in sigmas:
        print(sigma)
        tmp = get_nll(data, samples, sigma, batch_size=batch_size)
        tmp = np.mean(tmp)
        tmp = np.reshape(tmp, (1, 1))
        tmp = tf.squeeze(tmp, 1)
        lls = tf.concat([lls, tmp], 0)
        gc.collect()

    ind = np.argmax(lls)
    return sigmas[ind]

gen = decoder(z, training=False)
print('gen_image shape:', gen.shape)
# cross validate sigma
if sigma is None:
    # sigma_range = np.logspace(start = -1, stop = -0.3, num=20)
    sigma_range = np.logspace(start=-1, stop=-0.3, num=20)
    sigma = cross_validate_sigma(
        gen, x_valid, sigma_range, batch_size=BATCH_SIZE
    )
    sigma = sigma
else:
    sigma = float(sigma)
print("Using Sigma: {}".format(sigma))


def parzen(gen):
    '''parzen'''

    gc.collect()

    ll = get_nll(x_test, gen, sigma, batch_size=BATCH_SIZE)
    se = np.std(ll) / np.sqrt(x_test.shape[0])

    print("Log-Likelihood of test set = {}, se: {}".format(np.mean(ll), se))
    print(UNIQUE_RUN_ID)

    return np.mean(ll), se


mean_ll, se_ll = parzen(gen)
