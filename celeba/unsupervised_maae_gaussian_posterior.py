"""
unsupervised adversarial autoencoder on celeba.
 We are using:
    - Gaussian distribution as prior and posterior distribution.
    - convolution layers.
"""
import os
import time
from pathlib import Path
from math import sin, cos
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import randint
from scipy.linalg import sqrtm
from skimage.transform import resize

UNIQUE_RUN_ID = 'unsupervised_3d_maae_lam_gaussian_posterior_15z'
PROJECT_ROOT = Path.cwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# -------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------
output_dir = PROJECT_ROOT / 'outputs'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'unsupervised_3d_maae_lam_gaussian_posterior_15z'
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
NUM_OF_D = 3
n_samples = len(os.listdir(train_data))
z_dim = 15
h_dim = 128

ae_lr = 0.0002
gen_lr = 0.0001
dc_lr = 0.0001
# max_lr = 0.001
# step_size = 2 * np.ceil(n_samples / batch_size)
# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate
# WEIGHT_INIT_STDDEV = 0.02
# step_size = 2 * np.ceil(x_train.shape[0] / batch_size)
global_step = 0
n_epochs = 25
keep_pro = 0.1
# -------------------------------------------------------------------------------------------------------------
# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.
lam = [tf.Variable(tf.constant(0.01))]
# gen_loss_weight=[tf.Variable(1-tf.constant(1.))]
cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()
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
    for ind in range(NUM_OF_D):
        tf.keras.models.save_model(
            discriminator[ind],
            f'./{UNIQUE_RUN_ID}/discriminator{ind}_{epoch}.model',
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )


# -------------------------------------------------------------------------------------------------------------
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
    x = tf.keras.layers.Reshape((8, 8, 128))(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    reconstruction = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, padding="same",
                                                     activation='sigmoid')(
        x)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def make_discriminator_model(keep_pro):
    encoded = tf.keras.Input(shape=(1, 1, z_dim))
    x = tf.keras.layers.Dense(h_dim)(encoded)
    x = tf.keras.layers.ReLU(0.2)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.ReLU(keep_pro)(x)
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


def autoencoder_loss(inputs, reconstruction, ae_loss_weight):
    return ae_loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output):
    d_loss = [tf.reduce_mean(-tf.math.log(real_output[ind]) - tf.math.log(1 - fake_output[ind])) for ind in
              range(NUM_OF_D)]
    return d_loss


def mix_pre(G_losses):
    used_l = tf.nn.softplus(lam)
    weights = tf.exp(used_l * G_losses)

    denom = tf.reduce_sum(weights)
    weights = tf.math.divide(weights, denom)
    print('lam_weights shape', weights.shape)
    g_loss = tf.reduce_sum(weights * G_losses)
    return g_loss, used_l


def generator_loss(fake_output, gen_loss_weight):
    G_losses = [tf.reduce_mean(tf.math.log(1 - fake_output[ind])) for ind in
                range(NUM_OF_D)]

    # return tf.reduce_max(G_losses)
    G_losses, used_l = mix_pre(G_losses)
    return gen_loss_weight * G_losses, used_l


def generate_fake_images(decoder, NUM):
    """ Generate subplots with generated examples. """
    z = tf.random.normal([NUM, z_dim], mean=0.0, stddev=5.0)
    dec_input = np.reshape(z, (NUM, z_dim))

    x = decoder(dec_input.astype('float32'), training=False).numpy()

    x = np.array(x.reshape(NUM, img_size, img_size, num_c))

    return x


def gaussian_mixture(batch_size, n_labels, n_dim, x_var=5., y_var=5., label_indices=None):
    np.random.seed(0)
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0., x_var, (batch_size, (int)(n_dim / 2)))
    y = np.random.normal(0., y_var, (batch_size, (int)(n_dim / 2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim / 2)):
            if label_indices is not None:
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels),
                                                     n_labels)

    return z


def reparameterization(z_mean, z_std):
    # Probabilistic with Gaussian posterior distribution
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0., stddev=5.)
    z = z_mean + (1e-8 + z_std) * epsilon
    return z


# Define optimizers
ae_optimizer = tf.keras.optimizers.Adam(lr=ae_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=dc_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=gen_lr)
encoder = make_encoder_model()
decoder = make_decoder_model()
discriminator = [make_discriminator_model(0.1 + 0.1 * i) for i in range(NUM_OF_D)]
encoder.summary()
decoder.summary()


@tf.function
def train_step(batch_x):
    with tf.GradientTape(persistent=True) as ae_tape:
        z_mean, z_std = encoder(batch_x, training=True)
        z = reparameterization(z_mean, z_std)
        decoder_output = decoder(z, training=True)
        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))
    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape(persistent=True) as dc_tape:
        real_distribution = tf.random.normal([batch_x.shape[0] * NUM_OF_D, 1, 1, z_dim], mean=0.0, stddev=5.0)

        real_distribution = tf.split(real_distribution, NUM_OF_D)
        z_mean, z_std = encoder(batch_x, training=True)
        z = [reparameterization(z_mean, z_std) for _ in range(NUM_OF_D)]

        dc_real = [discriminator[ind](real_distribution[ind], training=True) for ind in
                   range(NUM_OF_D)]
        dc_fake = [discriminator[ind](z[ind], training=True) for ind in
                   range(NUM_OF_D)]

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake)

        # Discriminator Acc
        dc_acc = [accuracy(tf.concat([tf.ones_like(dc_real[ind]), tf.zeros_like(dc_fake[ind])], axis=0),
                           tf.concat([dc_real[ind], dc_fake[ind]], axis=0)) for ind in range(NUM_OF_D)]

    dc_grads = [dc_tape.gradient(dc_loss[ind], discriminator[ind].trainable_variables) for ind in range(NUM_OF_D)]
    for ind in range(NUM_OF_D):
        dc_optimizer.apply_gradients(zip(dc_grads[ind], discriminator[ind].trainable_variables))

    with tf.GradientTape(persistent=True) as gen_tape:
        z_mean, z_std = encoder(batch_x, training=True)
        z = reparameterization(z_mean, z_std)
        dc_fake = [discriminator[ind](z, training=True) for ind in
                   range(NUM_OF_D)]
        # Generator loss
        gen_loss, used_l = generator_loss(dc_fake, gen_loss_weight)
        sum_loss = gen_loss - 0.001 * used_l
    gen_grads = gen_tape.gradient(sum_loss, encoder.trainable_variables + lam)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables + lam))

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
        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_dc_loss_avg = tf.metrics.Mean()
        epoch_dc_acc_avg = tf.metrics.Mean()
        epoch_gen_loss_avg = tf.metrics.Mean()
        for batch, (batch_x) in enumerate(x_train):
            # -------------------------------------------------------------------------------------------------------------
            global_step = global_step + 1
            ae_loss, dc_loss, dc_acc, gen_loss, used_l = train_step(batch_x)

            epoch_ae_loss_avg(ae_loss)
            epoch_dc_loss_avg(dc_loss)
            epoch_dc_acc_avg(dc_acc)
            epoch_gen_loss_avg(gen_loss)

            if global_step in [100, 200, 400, 800, 1500]:
                # Sampling
                num = 5
                """ Generate subplots with generated examples. """
                z = tf.random.normal([num * num, 1, 1, z_dim], mean=0.0, stddev=5.)
                images = decoder(z, training=False)
                plt.figure(figsize=(num, num))
                for i in range(num * num):
                    # Get image and reshape
                    image = images[i]
                    plt.subplot(num, num, i + 1)
                    plt.imshow(image[:, :, :])
                    plt.axis('off')
                plt.savefig(style_dir / ('epoch_%d.png' % global_step))
            # write data after every 10-th iterations
            if global_step % 10 == 0:
                tf.summary.scalar("ae_loss", np.mean(epoch_ae_loss_avg.result()), global_step)
                tf.summary.scalar("dc_loss", np.mean(epoch_dc_loss_avg.result()), global_step)
                tf.summary.scalar("gen_loss", np.mean(epoch_gen_loss_avg.result()), global_step)
                tf.summary.scalar("dc_acc", np.mean(epoch_dc_acc_avg.result()), global_step)
                tf.summary.scalar("used_l", np.mean(used_l), global_step)

        epoch_time = time.time() - start
        print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f} ' \
              .format(epoch, epoch_time,
                      epoch_time * (n_epochs - epoch),
                      epoch_ae_loss_avg.result(),
                      epoch_dc_loss_avg.result(),
                      epoch_dc_acc_avg.result(),
                      epoch_gen_loss_avg.result()
                      ))
        #save models after 10 eopch
        if (epoch + 1) % 10 == 0:
            save_models(decoder, encoder, discriminator, epoch)
        # -------------------------------------------------------------------------------------------------------------
        if (epoch + 1) % 1 == 0:
            # Reconstruction image
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

            # Sampling
            num = 5
            """ Generate subplots with generated examples. """
            z = tf.random.normal([num * num, 1, 1, z_dim], mean=0.0, stddev=5.)
            images = decoder(z, training=False)
            plt.figure(figsize=(num, num))
            for i in range(num * num):
                # Get image and reshape
                # Plot
                plt.subplot(num, num, i + 1)
                plt.imshow(image[:, :, :])
                plt.axis('off')
            plt.savefig(style_dir / ('epoch_%d.png' % epoch))
