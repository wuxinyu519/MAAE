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
from math import sin, cos
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf


from sklearn.model_selection import train_test_split

from numpy.random import randint


from skimage.transform import resize


UNIQUE_RUN_ID = 'unsupervised_3d_maae_max_gaussian_posterior_dense_8z_test'
PROJECT_ROOT = Path.cwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# -------------------------------------------------------------------------------------------------------------
# Set random seed
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# -------------------------------------------------------------------------------------------------------------
output_dir = PROJECT_ROOT / 'outputs'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'unsupervised_3d_maae_max_gaussian_posterior_dense_8z_test'
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
NUM_OF_D = 3
n_samples = x_train.shape[0]
z_dim =8
h_dim = 1000

ae_lr = 0.0002
gen_lr = 0.0001
dc_lr = 0.0001
# max_lr = 0.001
# step_size = 2 * np.ceil(x_train.shape[0] / batch_size)
# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate
# WEIGHT_INIT_STDDEV = 0.02
# step_size = 2 * np.ceil(x_train.shape[0] / batch_size)
global_step = 0
n_epochs = 200
keep_pro = 0.1
# -------------------------------------------------------------------------------------------------------------
# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.
lam = [tf.Variable(tf.constant(0.1))]
# gen_loss_weight=[tf.Variable(1-tf.constant(1.))]
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()
# weight_init = tf.keras.initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV)
# prepare the inception v3 model
# model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape(x_train.shape[0], img_size * img_size * num_c)
x_test = x_test.reshape(x_test.shape[0], img_size * img_size * num_c)
x_test_fid = x_test
real_images = np.reshape(x_test_fid, (x_test_fid.shape[0], img_size, img_size, num_c))
# real_images = np.array(x_test_fid)
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
def make_encoder_model():
    inputs = tf.keras.Input(shape=(img_size * img_size * num_c,))
    kernel_initializer = tf.initializers.RandomNormal(0.0, 0.01)
    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)
    mean = tf.keras.layers.Dense(z_dim, kernel_initializer=kernel_initializer)(x)
    stddev = tf.keras.layers.Dense(z_dim, kernel_initializer=kernel_initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=[mean, stddev])
    return model


def make_decoder_model():
    encoded = tf.keras.Input(shape=(z_dim,))
    kernel_initializer = tf.initializers.RandomNormal(0.0, 0.01)
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


def make_discriminator_model(pro):
    encoded = tf.keras.Input(shape=(z_dim,))
    kernel_initializer = tf.initializers.RandomNormal(0.0, 0.01)

    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(encoded)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dense(h_dim, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Dropout(pro)(x)
    x = tf.keras.layers.ReLU()(x)
    prediction = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


def autoencoder_loss(inputs, reconstruction):
    return ae_loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output, dc_loss_weight):
    d_loss = [tf.reduce_mean(-tf.math.log(real_output[ind]) - tf.math.log(1 - fake_output[ind])) for ind in
              range(NUM_OF_D)]
    # loss_real = [cross_entropy(tf.ones_like(real_output[ind]), real_output[ind]) for ind in range(NUM_OF_D)]
    # loss_fake = [cross_entropy(tf.zeros_like(fake_output[ind]), fake_output[ind]) for ind in range(NUM_OF_D)]
    # d_loss = [loss_real[i] + loss_fake[i] for i in range(NUM_OF_D)]
    return d_loss
    # return tf.reduce_mean(-tf.math.log(real_output) - tf.math.log(1 - fake_output))


def mix_pre(G_losses):
    used_l = tf.nn.softplus(lam)
    weights = tf.exp(used_l*G_losses)
    denom = tf.reduce_sum(weights)
    weights = tf.math.divide(weights, denom)
    print('lam_weights shape', weights.shape)
    g_loss = tf.reduce_sum(weights * G_losses)
    return g_loss,used_l


def generator_loss(fake_output, gen_loss_weight):
    # G_losses = [cross_entropy(tf.ones_like(fake_output[ind]), fake_output[ind]) for ind in range(NUM_OF_D)]
    G_losses =   [tf.reduce_mean(tf.math.log(1-fake_output[ind])) for ind in
                range(NUM_OF_D)]
    # g_losses,used_l= mix_pre(G_losses)
    # return gen_loss_weight * g_losses,used_l
    return tf.reduce_max(G_losses)


def generate_fake_images(decoder, NUM):
    """ Generate subplots with generated examples. """
    z = tf.random.normal([NUM, z_dim], mean=0.0, stddev=5.0)
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


def gaussian_mixture(batch_size, n_labels, n_dim, x_var=5., y_var=1., label_indices=None):
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
    z = z_mean + tf.exp(z_std * .5) * epsilon
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
discriminator = [make_discriminator_model(0.2+0.1*i) for i in range(NUM_OF_D)]
encoder.summary()
decoder.summary()


@tf.function
def train_step(batch_x,batch_y):
    with tf.GradientTape(persistent=True) as ae_tape:
        z_mean, z_std = encoder(batch_x, training=True)

        z = reparameterization(z_mean, z_std)

        decoder_output = decoder(z, training=True)
        # Autoencoder loss

        ae_loss = autoencoder_loss(batch_x, decoder_output)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape(persistent=True) as dc_tape:
        real_distribution = tf.random.normal([batch_x.shape[0]*NUM_OF_D, z_dim], mean=0.0, stddev=5.0)


        real_distribution = tf.split(real_distribution, NUM_OF_D)
        z_mean, z_std = encoder(batch_x, training=True)

        z = reparameterization(z_mean, z_std)
        dc_real = [discriminator[ind](real_distribution[ind], training=True) for ind in
                   range(NUM_OF_D)]
        dc_fake = [discriminator[ind](z, training=True) for ind in
                   range(NUM_OF_D)]

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)
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
        gen_loss= generator_loss(dc_fake, gen_loss_weight)
        # sum_loss = gen_loss - 0.001 * used_l
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
    # real_images = np.array(generate_real_images(x_test_fid, 10000))
    enc_std = []
    enc_dec_std = []
    all_std = []
    for epoch in range(n_epochs):
        start = time.time()

        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_dc_loss_avg = tf.metrics.Mean()
        epoch_dc_acc_avg = tf.metrics.Mean()
        epoch_gen_loss_avg = tf.metrics.Mean()
        for batch, (batch_x, batch_y) in enumerate(train_dataset):
            # -------------------------------------------------------------------------------------------------------------
            # Calculate cyclic learning rate
            global_step = global_step + 1

            ae_loss, dc_loss, dc_acc, gen_loss= train_step(batch_x,batch_y)

            epoch_ae_loss_avg(ae_loss)
            # epoch_dc_loss_avg(dc_loss)
            # epoch_dc_acc_avg(dc_acc)
            epoch_gen_loss_avg(gen_loss)

            # caculate the Standard Deviation of the encoder loss(recon+adversarial) after each 100 iterations.
            enc_std.append(np.mean(gen_loss))
            enc_dec_std.append(np.mean(ae_loss))
            all_std.append(np.mean(gen_loss) + np.mean(ae_loss))
            if (batch+1) % 100==0:
                tf.summary.scalar("enc_std", np.mean(np.std(enc_std)), global_step)
                tf.summary.scalar("enc_dec_std", np.mean(np.std(enc_dec_std)), global_step)
                tf.summary.scalar("all_std", np.mean(np.std(all_std)), global_step)
                enc_std.clear()
                enc_dec_std.clear()
                all_std.clear()
            if global_step % 10 == 0:
                tf.summary.scalar("ae_loss", np.mean(epoch_ae_loss_avg.result()), global_step)
                tf.summary.scalar("dc_loss", np.mean(dc_loss), global_step)
                # [tf.summary.scalar("dc_loss%d" % ind, np.mean(dc_loss[ind]), global_step) for ind in range(NUM_OF_D)]
                tf.summary.scalar("gen_loss", np.mean(epoch_gen_loss_avg.result()), global_step)
                # [tf.summary.scalar("gen_loss%d" % ind, np.mean(G_losses[ind]), global_step) for ind in range(NUM_OF_D)]
                tf.summary.scalar("dc_acc", np.mean(dc_acc), global_step)
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
        # if (epoch + 1) % 350 == 0:
        #     # fid score
        #     fake_images = np.array(generate_fake_images(decoder, x_test.shape[0]))
        #
        #     # resize images
        #     images1 = scale_images(real_images, (299, 299, 3))
        #     images2 = scale_images(fake_images, (299, 299, 3))
        #     images1 = preprocess_input(images1)
        #     images2 = preprocess_input(images2)
        #     print('Scaled', images1.shape, images2.shape)
        #
        #     fid = calculate_fid(model, images1, images2)
        #
        #     tf.summary.scalar("fid_score", np.mean(fid), epoch)
        #     # fid scores
        #     print('FID (different): %.3f' % fid)
        if (epoch+1)  % 10 == 0:
            save_models(decoder, encoder, discriminator, epoch)
        # -------------------------------------------------------------------------------------------------------------
        if epoch % 1 == 0:
            # Latent Space
            z_mean, z_std = encoder(x_test, training=False)
            z = reparameterization(z_mean, z_std)
            # epsilon = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
            # z = z_mean + (1e-8 + z_std) * epsilon
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
            z_mean, z_std = encoder(x_test[:n_digits], training=False)
            z = reparameterization(z_mean, z_std)
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
            num = 5
            """ Generate subplots with generated examples. """
            z = tf.random.normal([num * num, z_dim], mean=0.0, stddev=5.)
            images = decoder(z, training=False)
            plt.figure(figsize=(num, num))
            for i in range(num * num):
                # Get image and reshape
                image = images[i]
                image = np.reshape(image, (img_size, img_size, num_c))
                # Plot
                plt.subplot(num, num, i + 1)
                plt.imshow(image[:, :, 0], cmap='gray')
                plt.axis('off')
            plt.savefig(style_dir / ('epoch_%d.png' % epoch))
    print(UNIQUE_RUN_ID)