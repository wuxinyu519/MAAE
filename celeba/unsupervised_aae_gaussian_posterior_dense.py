"""
Probabilistic unsupervised adversarial autoencoder.
 We are using:
    - Gaussian distribution as prior and posterior distribution.
    - Dense layers.
    - Cyclic learning rate.
"""
import os
import time
from pathlib import Path
from math import sin, cos, sqrt
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from random import *
from scipy.linalg import sqrtm
import pandas as pd
import keras.backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from numpy.random import randint
from scipy.linalg import sqrtm

from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from skimage.color import gray2rgb
from tqdm import tqdm

UNIQUE_RUN_ID = 'unsupervised_aae_gaussian_posterior_cnn_15z'
PROJECT_ROOT = Path.cwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# -------------------------------------------------------------------------------------------------------------
# Set random seed
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# -------------------------------------------------------------------------------------------------------------
output_dir = PROJECT_ROOT / 'outputs'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'unsupervised_aae_gaussian_posterior_cnn_15z'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'latent_space'
latent_space_dir.mkdir(exist_ok=True)

reconstruction_dir = experiment_dir / 'reconstruction'
reconstruction_dir.mkdir(exist_ok=True)

style_dir = experiment_dir / 'style'
style_dir.mkdir(exist_ok=True)

train_data = PROJECT_ROOT / 'celeba' / 'img_align_celeba'
test_data = PROJECT_ROOT / 'celeba' / 'test_data'

# -------------------------------------------------------------------------------------------------------------
# HYPERPARAMETER
# -----------------------------------------------------------------------------------------
img_size = 32
num_c = 3
batch_size = 100
n_samples = len(os.listdir(train_data))
z_dim = 15
h_dim = 128
# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate
ae_lr = 0.0002
gen_lr = 0.0001
dc_lr = 0.0001
# max_lr = 0.001
# step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0
n_epochs = 25

# -------------------------------------------------------------------------------------------------------------
# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.
# lam=[tf.Variable(tf.constant(1.))]

cross_entropy = tf.keras.losses.BinaryCrossentropy()
# cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits()
mse = tf.keras.losses.MeanSquaredError()
# mse=tf.math.squared_difference()
accuracy = tf.keras.metrics.BinaryAccuracy()
# weight_init = tf.keras.initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV)


# -------------------------------------------------------------------------------------------------------------
# Loading data
# -------------------------------------------------------------------------------------------------------------
print("Loading data...")

x_train = tf.keras.preprocessing.image_dataset_from_directory(
    train_data,
    label_mode=None,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
    seed=42,
)

x_train = x_train.map(lambda x: x / 255.0)
x_train = x_train.prefetch(buffer_size=10 * batch_size)
x_test = tf.keras.preprocessing.image_dataset_from_directory(
    test_data,
    label_mode=None,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False
)
x_test = x_test.map(lambda x: x / 255.0)


def save_models(decoder, encoder, discriminator, epoch):
    """ Save models at specific point in time. """
    tf.keras.models.save_model(
        decoder,
        f'./{UNIQUE_RUN_ID}/decoder_{epoch}.model',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    tf.keras.models.save_model(
        encoder,
        f'./{UNIQUE_RUN_ID}/encoder_{epoch}.model',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    tf.keras.models.save_model(
        discriminator,
        f'./{UNIQUE_RUN_ID}/discriminator_{epoch}.model',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )


# -------------------------------------------------------------------------------------------------------------
# Create models
def make_encoder_model():
    inputs = tf.keras.Input(shape=(img_size, img_size, num_c,))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding="same")(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    mean = tf.keras.layers.Dense(z_dim)(x)
    mean = tf.keras.layers.Reshape((1, 1, z_dim))(mean)
    stddev = tf.keras.layers.Dense(z_dim, activation='softplus')(x)
    stddev = tf.keras.layers.Reshape((1, 1, z_dim))(stddev)
    model = tf.keras.Model(inputs=inputs, outputs=[mean, stddev])
    return model


def make_decoder_model():
    encoded = tf.keras.Input(shape=(1, 1, z_dim))
    x = tf.keras.layers.Dense(128 * 8 * 8)(encoded)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Reshape((8, 8, 128))(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)


    reconstruction = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, padding="same", activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def make_discriminator_model():
    encoded = tf.keras.Input(shape=(1, 1, z_dim))

    x = tf.keras.layers.Dense(h_dim)(encoded)
    x = tf.keras.layers.ReLU(0.2)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.ReLU(0.2)(x)
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


def autoencoder_loss(inputs, reconstruction, ae_loss_weight):
    return ae_loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output, dc_loss_weight):
    # loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    # loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # return dc_loss_weight * tf.reduce_mean(loss_fake + loss_real)
    return tf.reduce_mean(-tf.math.log(real_output) - tf.math.log(1 - fake_output))


def generator_loss(fake_output, gen_loss_weight):
    # return gen_loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)
    return tf.reduce_mean(tf.math.log(1-fake_output))


def generate_fake_images(decoder, NUM):
    """ Generate subplots with generated examples. """
    z = tf.random.normal([NUM, z_dim], mean=0.0, stddev=5.)
    dec_input = np.reshape(z, (NUM, z_dim))

    x = decoder(dec_input.astype('float32'), training=False).numpy()

    x = np.array(x.reshape(NUM, img_size, img_size, num_c))

    return x


# scale an array of images to a new size
# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def gaussian_mixture(batch_size, labels, n_classes):
    x_stddev = 0.5
    y_stddev = 0.1
    shift = 3 * x_stddev

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


def reparameterization(z_mean, z_std):
    # Probabilistic with Gaussian posterior distribution
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0., stddev=5.)
    z = z_mean + (1e-8 + z_std) * epsilon
    return z


# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate

# ae_lr = 0.0002
# dc_lr = 0.0001
# gen_lr = 0.0001
# Define optimizers
ae_optimizer = tf.keras.optimizers.Adam(lr=ae_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=dc_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=gen_lr)
encoder = make_encoder_model()
decoder = make_decoder_model()
discriminator = make_discriminator_model()
encoder.summary()
decoder.summary()
discriminator.summary()


@tf.function
def train_step(batch_x):
    with tf.GradientTape(persistent=True) as ae_tape:
        z_mean, z_std = encoder(batch_x, training=True)
        # Probabilistic with Gaussian posterior distribution
        z = reparameterization(z_mean, z_std)

        # decoder_output = decoder(z, training=True)
        decoder_output = decoder(z, training=True)

        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape(persistent=True) as dc_tape:
        real_distribution = tf.random.normal([batch_x.shape[0], 1, 1, z_dim], mean=0.0, stddev=5.)
        z_mean, z_std = encoder(batch_x, training=True)
        z = reparameterization(z_mean, z_std)
        # labels = np.random.randint(0, n_labels, size=batch_size)
        # real_distribution = gaussian_mixture(batch_size=batch_size,labels=labels,n_classes=n_labels)
        # epsilon = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
        # z = z_mean + (1e-8 + z_std) * epsilon
        # z=reparameterization(z_mean, z_std)
        dc_real = discriminator(real_distribution, training=True)
        dc_fake = discriminator(z, training=True)

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                          tf.concat([dc_real, dc_fake], axis=0))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)

    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    with tf.GradientTape(persistent=True) as gen_tape:
        z_mean, z_std = encoder(batch_x, training=True)
        z = reparameterization(z_mean, z_std)
        dc_fake = discriminator(z, training=True)
        # Generator loss
        gen_loss = generator_loss(dc_fake, gen_loss_weight)

    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

    del ae_tape, dc_tape, gen_tape
    return ae_loss, dc_loss, dc_acc, gen_loss


# -------------------------------------------------------------------------------------------------------------
# Training loop
if not os.path.exists(f'./cruves/{UNIQUE_RUN_ID}'):
    os.mkdir(f'./cruves/{UNIQUE_RUN_ID}')
writer = tf.summary.create_file_writer(f'./cruves/{UNIQUE_RUN_ID}/')
with writer.as_default():
    for epoch in range(n_epochs):
        start = time.time()
        # if epoch in [60, 100, 300]:
        #     ae_lr = ae_lr / 2
        #     gen_lr = gen_lr / 2
        #     dc_lr = dc_lr / 2
        #     max_lr = max_lr / 2
        #     step_size = step_size / 2
        #
        #     print('learning rate changed!')

        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_dc_loss_avg = tf.metrics.Mean()
        epoch_dc_acc_avg = tf.metrics.Mean()
        epoch_gen_loss_avg = tf.metrics.Mean()
        for batch, (batch_x) in enumerate(x_train):
            # -------------------------------------------------------------------------------------------------------------
            # Calculate cyclic learning rate
            global_step = global_step + 1
            # cycle = np.floor(1 + global_step / (2 * step_size))
            # x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
            # ae_clr = ae_lr + (max_lr - ae_lr) * max(0, 1 - x_lr)
            # dc_clr = dc_lr + (max_lr - dc_lr) * max(0, 1 - x_lr)
            # gen_clr = gen_lr + (max_lr - gen_lr) * max(0, 1 - x_lr)
            # ae_optimizer.lr = ae_clr
            # dc_optimizer.lr = dc_clr
            # gen_optimizer.lr = gen_clr
            ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x)

            epoch_ae_loss_avg(ae_loss)
            epoch_dc_loss_avg(dc_loss)
            epoch_dc_acc_avg(dc_acc)
            epoch_gen_loss_avg(gen_loss)

            # write data after every n-th batch
            if global_step in[100,200,400,800,1500]:
                # Sampling
                num = 5
                """ Generate subplots with generated examples. """
                z = tf.random.normal([num * num, 1, 1, z_dim], mean=0.0, stddev=5.)
                images = decoder(z, training=False)
                plt.figure(figsize=(num, num))
                for i in range(num * num):
                    # Get image and reshape
                    image = images[i]
                    # image = np.reshape(image, (img_size, img_size, num_c))
                    # Plot
                    plt.subplot(num, num, i + 1)
                    plt.imshow(image[:, :, :])
                    plt.axis('off')
                plt.savefig(style_dir / ('epoch_%d.png' % global_step))
            if global_step % 10 == 0:
                tf.summary.scalar("ae_loss", np.mean(epoch_ae_loss_avg.result()), global_step)
                tf.summary.scalar("dc_loss", np.mean(epoch_dc_loss_avg.result()), global_step)
                tf.summary.scalar("gen_loss", np.mean(epoch_gen_loss_avg.result()), global_step)
                tf.summary.scalar("dc_acc", np.mean(epoch_dc_acc_avg.result()), global_step)

        epoch_time = time.time() - start
        print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f} ' \
              .format(epoch, epoch_time,
                      epoch_time * (n_epochs - epoch),
                      epoch_ae_loss_avg.result(),
                      epoch_dc_loss_avg.result(),
                      epoch_dc_acc_avg.result(),
                      epoch_gen_loss_avg.result()
                      ))

        if (epoch + 1) % 10 == 0:
            save_models(decoder, encoder, discriminator, epoch)
        # -------------------------------------------------------------------------------------------------------------
        if (epoch + 1) % 1 == 0:
            # Reconstruction
            n_digits = 20  # how many digits we will display
            for batch, (batch_x) in enumerate(x_test):
                test_images = batch_x[:n_digits]
                break

            z_mean, z_std = encoder(test_images, training=False)
            z = reparameterization(z_mean, z_std)
            x_test_decoded = decoder(z, training=False)

            x_test_decoded = np.reshape(x_test_decoded, [-1, img_size, img_size, num_c])
            fig = plt.figure(figsize=(20, 4))
            for i in range(n_digits):
                # display original
                ax = plt.subplot(2, n_digits, i + 1)
                plt.imshow(test_images[i])
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, n_digits, i + 1 + n_digits)
                plt.imshow(x_test_decoded[i][:, :, :])
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.savefig(reconstruction_dir / ('epoch_%d.png' % epoch))
            plt.close('all')

            # Conditioned Sampling
            # Sampling
            num = 5
            """ Generate subplots with generated examples. """
            z = tf.random.normal([num * num, 1, 1, z_dim], mean=0.0, stddev=5.)
            images = decoder(z, training=False)
            plt.figure(figsize=(num, num))
            for i in range(num * num):
                # Get image and reshape
                image = images[i]
                # image = np.reshape(image, (img_size, img_size, num_c))
                # Plot
                plt.subplot(num, num, i + 1)
                plt.imshow(image[:, :, :])
                plt.axis('off')
            plt.savefig(style_dir / ('epoch_%d.png' % epoch))
