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

from sklearn.model_selection import train_test_split

from numpy.random import randint


UNIQUE_RUN_ID = 'supervised_aae_mixture_posterior_dense_2z'
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

experiment_dir = output_dir / 'supervised_aae_mixture_posterior_dense_2z'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'latent_space'
latent_space_dir.mkdir(exist_ok=True)

reconstruction_dir = experiment_dir / 'reconstruction'
reconstruction_dir.mkdir(exist_ok=True)

style_dir = experiment_dir / 'style'
style_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------------------------------------------------------
# Loading data
# -------------------------------------------------------------------------------------------------------------
print("Loading data...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train,  # 把上面剩余的 x_train, y_train继续拿来切
    test_size=1 / 60  # test_size默认是0.25
)
# -------------------------------------------------------------------------------------------------------------
# HYPERPARAMETER
# -----------------------------------------------------------------------------------------
img_size = 28
num_c = 1
batch_size = 100
train_buf = x_train.shape[0]
n_labels = 10
n_samples = x_train.shape[0]
z_dim = 2
h_dim = 1000
# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate
ae_lr = 0.0002
gen_lr = 0.0001
dc_lr = 0.0001

global_step = 0
n_epochs = 20

# -------------------------------------------------------------------------------------------------------------
# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.

mse = tf.keras.losses.MeanSquaredError()

accuracy = tf.keras.metrics.BinaryAccuracy()

# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape(x_train.shape[0], img_size * img_size * num_c)
x_test = x_test.reshape(x_test.shape[0], img_size * img_size * num_c)
x_test_fid = x_test
real_images = np.reshape(x_test_fid, (x_test_fid.shape[0], img_size, img_size, num_c))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)


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
    inputs = tf.keras.Input(shape=(img_size * img_size * num_c,))
    kernel_initializer = tf.initializers.RandomNormal()
    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)
    z = tf.keras.layers.Dense(z_dim, kernel_initializer=kernel_initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=z)
    return model


def make_decoder_model():
    encoded = tf.keras.Input(shape=(z_dim,))
    kernel_initializer = tf.initializers.RandomNormal()
    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(encoded)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)
    reconstruction = tf.keras.layers.Dense(img_size * img_size * num_c, kernel_initializer=kernel_initializer,
                                           activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def make_discriminator_model():
    encoded = tf.keras.Input(shape=(z_dim + n_labels,))
    kernel_initializer = tf.initializers.RandomNormal()
    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(encoded)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)
    prediction = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


def autoencoder_loss(inputs, reconstruction, ae_loss_weight):
    return ae_loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(-tf.math.log(real_output) - tf.math.log(1 - fake_output))


def generator_loss(fake_output):
    return tf.reduce_mean(tf.math.log(1 - fake_output))


def generate_fake_images(decoder, NUM):
    """ Generate subplots with generated examples. """
    z = tf.random.normal([NUM, z_dim], mean=0.0, stddev=5.0)
    dec_input = np.reshape(z, (NUM, z_dim))

    x = decoder(dec_input.astype('float32'), training=False).numpy()

    x = np.array(x.reshape(NUM, img_size, img_size, num_c))

    return x


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


def reparameterization(z_mean, z_std):
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0., stddev=5.)
    z = z_mean + tf.exp(z_std * .5) * epsilon

    return z


# -------------------------------------------------------------------------------------------------------------
# Define learning rate
ae_optimizer = tf.keras.optimizers.Adam(lr=ae_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=dc_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=gen_lr)
encoder = make_encoder_model()
decoder = make_decoder_model()
discriminator = make_discriminator_model()
encoder.summary()
decoder.summary()


@tf.function
def train_step(batch_x, batch_y):
    with tf.GradientTape(persistent=True) as ae_tape:
        z = encoder(batch_x, training=True)
        decoder_output = decoder(z, training=True)
        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)
    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape(persistent=True) as dc_tape:
        z = encoder(batch_x, training=True)

        label_sample = np.random.randint(0, n_labels, size=[z.shape[0]])
        label_sample_one_hot = tf.one_hot(label_sample, n_labels)
        real_distribution = tf.random.normal(z.shape, mean=0, stddev=5)
        real_distribution_label = tf.concat(
            [real_distribution, label_sample_one_hot], axis=1
        )
        batch_y_one_hot = tf.one_hot(batch_y, n_labels)
        fake_distribution_label = tf.concat([z, batch_y_one_hot], axis=1)
        dc_real = discriminator(real_distribution_label, training=True)
        dc_fake = discriminator(fake_distribution_label, training=True)

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                          tf.concat([dc_real, dc_fake], axis=0))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)

    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    with tf.GradientTape(persistent=True) as gen_tape:
        z = encoder(batch_x, training=True)
        z_label = tf.concat([z, batch_y_one_hot], axis=1)
        dc_fake = discriminator(z_label, training=True)
        # Generator loss
        gen_loss = generator_loss(dc_fake)

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

        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_dc_loss_avg = tf.metrics.Mean()
        epoch_dc_acc_avg = tf.metrics.Mean()
        epoch_gen_loss_avg = tf.metrics.Mean()
        for batch, (batch_x, batch_y) in enumerate(train_dataset):
            # -------------------------------------------------------------------------------------------------------------
            global_step = global_step + 1
            ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, batch_y)

            epoch_ae_loss_avg(ae_loss)
            epoch_dc_loss_avg(dc_loss)
            epoch_dc_acc_avg(dc_acc)
            epoch_gen_loss_avg(gen_loss)

            # write data after every n-th batch

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

        save_models(decoder, encoder, discriminator, epoch)
        # -------------------------------------------------------------------------------------------------------------
        if (epoch) % 1 == 0:
            # Latent Space
            z = encoder(x_test, training=False)
            # z = reparameterization(z_mean, z_std)
            label_list = list(y_test)

            fig = plt.figure()
            classes = set(label_list)
            colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
            kwargs = {'alpha': 0.8, 'c': [colormap[i] for i in label_list]}
            ax = plt.subplot(111, aspect='equal')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            handles = [mpatches.Circle((0, 0), label=class_, color=colormap[i])
                       for i, class_ in enumerate(classes)]
            ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                      fancybox=True, loc='center left')
            plt.scatter(z[:, 0], z[:, 1], s=2, **kwargs)
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])

            plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
            plt.close('all')

            # Reconstruction
            n_digits = 20  # how many digits we will display
            z = encoder(x_test[:n_digits], training=False)
            x_test_decoded = decoder(z, training=False)
            x_test_decoded = np.reshape(x_test_decoded, [-1, img_size, img_size, num_c])
            fig = plt.figure(figsize=(20, 4))
            for i in range(n_digits):
                # display original
                ax = plt.subplot(2, n_digits, i + 1)
                plt.imshow(x_test[i].reshape(img_size, img_size))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, n_digits, i + 1 + n_digits)
                plt.imshow(x_test_decoded[i][:, :, 0])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.savefig(reconstruction_dir / ('epoch_%d.png' % epoch))
            plt.close('all')

            # Sampling
            nx, ny = 10, 10

            plt.subplot()
            gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
            i = 0
            for t in range(nx):
                for r in range(ny):
                    label = np.random.randint(t, t + 1, size=[1])
                    label_sample_one_hot = tf.one_hot(label, n_labels)
                    real_distribution = tf.random.normal(shape=[1, z_dim], mean=0, stddev=5)

                    x = decoder(real_distribution, training=False).numpy()
                    ax = plt.subplot(gs[i])
                    i += 1
                    img = np.array(x.tolist()).reshape(28, 28)
                    ax.imshow(img, cmap='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('auto')

            plt.savefig(style_dir / ('epoch_%d.png' % epoch))
            plt.close()
    print(UNIQUE_RUN_ID)
