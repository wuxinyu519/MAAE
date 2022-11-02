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
import tensorflow_probability as tfp
UNIQUE_RUN_ID = 'supervised_3d_maae_max_mixture_posterior_dense_2z'
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

experiment_dir = output_dir / 'supervised_3d_maae_max_mixture_posterior_dense_2z'
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
    x_train, y_train,
    test_size=1 / 60
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
ae_lr = 0.0001
gen_lr = 0.0002
dc_lr = 0.0002
NUM_OF_D=3
global_step = 0
n_epochs = 20

# -------------------------------------------------------------------------------------------------------------
# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.
# lam=[tf.Variable(tf.constant(1.))]
lam = [tf.Variable(tf.constant(0.1))]
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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
# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator

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


def make_discriminator_model(keep_pro):
    encoded = tf.keras.Input(shape=(z_dim+n_labels,))
    kernel_initializer = tf.initializers.RandomNormal()
    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(encoded)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Dropout(keep_pro)(x)
    x = tf.keras.layers.ReLU()(x)
    prediction = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer)(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


def autoencoder_loss(inputs, reconstruction, ae_loss_weight):
    return ae_loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output, dc_loss_weight):
    loss_real = [cross_entropy(tf.ones_like(real_output[ind]), real_output[ind]) for ind in range(NUM_OF_D)]
    loss_fake = [cross_entropy(tf.zeros_like(fake_output[ind]), fake_output[ind]) for ind in range(NUM_OF_D)]
    d_loss = [loss_real[i] + loss_fake[i] for i in range(NUM_OF_D)]
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
    G_losses = [cross_entropy(tf.ones_like(fake_output[ind]), fake_output[ind]) for ind in range(NUM_OF_D)]
    return tf.reduce_max(G_losses)


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

    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=[0,0],  # shape: [2, 3]
        scale_diag=[5.,1.])
    epsilon = mvn.sample(z_mean.shape[0])
    z = z_mean + tf.math.exp(0.5 * z_std) * epsilon


    return z
# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate

ae_optimizer = tf.keras.optimizers.Adam(lr=ae_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=dc_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=gen_lr)
encoder = make_encoder_model()
decoder = make_decoder_model()
keep_pro=0.1
discriminator =[ make_discriminator_model(keep_pro + 0.1*i) for i in range(NUM_OF_D)]
encoder.summary()
decoder.summary()





@tf.function
def train_step(batch_x,batch_y):

    with tf.GradientTape(persistent=True) as ae_tape:

        z= encoder(batch_x, training=True)

        decoder_output = decoder(z, training=True)
        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)
    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape(persistent=True) as dc_tape:

        z = encoder(batch_x, training=True)

        fake_distribution_label = tf.concat([z, tf.one_hot(batch_y, n_labels)], axis=1)

        #real z
        label_sample=np.random.randint(0, n_labels, size=[batch_x.shape[0]])

        label_sample_one_hot=tf.one_hot(label_sample, n_labels)
        real_distribution=gaussian_mixture(batch_x.shape[0], label_sample, n_labels)

        real_distribution_label=tf.concat(
            [real_distribution, label_sample_one_hot], axis=1
        )
        # real_distribution_label = tf.split(real_distribution_label, NUM_OF_D)
        dc_real = [discriminator[ind](real_distribution_label, training=True)for ind in range(NUM_OF_D)]
        dc_fake = [discriminator[ind](fake_distribution_label, training=True)for ind in range(NUM_OF_D)]

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

        # Discriminator Acc
        dc_acc = [accuracy(tf.concat([tf.ones_like(dc_real[ind]), tf.zeros_like(dc_fake[ind])], axis=0),
                           tf.concat([dc_real[ind], dc_fake[ind]], axis=0)) for ind in range(NUM_OF_D)]

    dc_grads = [dc_tape.gradient(dc_loss[ind], discriminator[ind].trainable_variables) for ind in range(NUM_OF_D)]
    for ind in range(NUM_OF_D):
        dc_optimizer.apply_gradients(zip(dc_grads[ind], discriminator[ind].trainable_variables))

    with tf.GradientTape(persistent=True) as gen_tape:
        z = encoder(batch_x, training=True)
        fake_distribution_label = tf.concat([z, tf.one_hot(batch_y, n_labels)], axis=1)
        dc_fake = [discriminator[ind](fake_distribution_label, training=True)for ind in range(NUM_OF_D)]
        # Generator loss
        gen_loss= generator_loss(dc_fake, gen_loss_weight)
        # sum_loss=gen_loss-0.001*used_l
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
        if epoch in [30,80]:
            dc_lr=dc_lr/2
            gen_lr=gen_lr/2
            ae_lr=ae_lr/2

        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_dc_loss_avg = tf.metrics.Mean()
        epoch_dc_acc_avg = tf.metrics.Mean()
        epoch_gen_loss_avg = tf.metrics.Mean()
        for batch, (batch_x,batch_y) in enumerate(train_dataset):
            # -------------------------------------------------------------------------------------------------------------
            # Calculate cyclic learning rate
            global_step = global_step + 1
            gen_optimizer.lr = gen_lr
            dc_optimizer.lr = dc_lr
            ae_optimizer.lr = ae_lr
            ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, batch_y)

            epoch_ae_loss_avg(ae_loss)
            epoch_gen_loss_avg(gen_loss)

            # write data after every n-th batch

            if global_step % 10 == 0:
                tf.summary.scalar("ae_loss", np.mean(epoch_ae_loss_avg.result()), global_step)
                [tf.summary.scalar("dc_loss%d"%ind, np.mean(dc_loss[ind]), global_step)for ind in range(NUM_OF_D)]
                tf.summary.scalar("gen_loss", np.mean(epoch_gen_loss_avg.result()), global_step)
                [tf.summary.scalar("dc_acc%d"%ind, np.mean(dc_acc[ind]), global_step)for ind in range(NUM_OF_D)]
                # tf.summary.scalar("used_l", np.mean(used_l), global_step)
        epoch_time = time.time() - start
        print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f} ' \
              .format(epoch, epoch_time,
                      epoch_time * (n_epochs - epoch),
                      epoch_ae_loss_avg.result(),
                      np.mean(dc_loss),
                      np.mean(dc_acc),
                      epoch_gen_loss_avg.result()
                      ))

        save_models(decoder, encoder, discriminator, epoch)
        # -------------------------------------------------------------------------------------------------------------
        if (epoch) % 1 == 0:
            # Latent Space
            z= encoder(x_test, training=False)
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
            z= encoder(x_test[:n_digits], training=False)

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
                    real_distribution = gaussian_mixture(1, label, n_labels)
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

