# Import
import time

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np

from pathlib import Path
from matplotlib import gridspec
# Initialize variables
NUM_EPOCHS = 25
img_size=28
num_c=1
BUFFER_SIZE = 60000
BATCH_SIZE = 100
ITERATIONS = int(BUFFER_SIZE / BATCH_SIZE)
NOISE_DIMENSION = 75
UNIQUE_RUN_ID = str('gman_lam_75z')
PRINT_STATS_AFTER_BATCH = 10
SAVE_MODEL_AFTER_BATCH = 5
OPTIMIZER_LR = 0.0002
OPTIMIZER_BETAS = (0.5, 0.999)
WEIGHT_INIT_STDDEV = 0.02
NUM_OF_D = 5
lam = [tf.Variable(initial_value=0.01, name='lambda')]
PROJECT_ROOT = Path.cwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Initialize loss function, init schema and optimizers
weight_init = tf.keras.initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV)
generator_optimizer = tf.keras.optimizers.Adam(OPTIMIZER_LR,
                                                       beta_1=OPTIMIZER_BETAS[0], beta_2=OPTIMIZER_BETAS[1])
discriminator_optimizer = tf.keras.optimizers.Adam(OPTIMIZER_LR,
                                                           beta_1=OPTIMIZER_BETAS[0], beta_2=OPTIMIZER_BETAS[1])
accuracy = tf.keras.metrics.BinaryAccuracy()
# Make run directory
output_dir = PROJECT_ROOT / 'outputs'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'gman_lam_75z'
experiment_dir.mkdir(exist_ok=True)

fake_dir = experiment_dir / 'fake_img'
fake_dir.mkdir(exist_ok=True)

style_dir = experiment_dir / 'style'
style_dir.mkdir(exist_ok=True)


def generate_image(generator, epoch=0):
    """ Generate subplots with generated examples. """
    noise = generate_noise(BATCH_SIZE)
    images = generator(noise, training=False)
    plt.figure(figsize=(10, 10))
    for i in range(100):
        # Get image and reshape
        image = images[i]
        image = np.reshape(image, (img_size, img_size,num_c))
        # Plot
        plt.subplot(10, 10, i + 1)
        plt.imshow(image[:,:,0], cmap='gray')
        plt.axis('off')
    plt.savefig(fake_dir / ('epoch_%d.png' % epoch))


def load_data():
    """ Load data """
    (images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    images = images.reshape(images.shape[0], img_size, img_size, num_c)
    images = images.astype('float32')
    # images = (images - 127.5) / 127.5
    images = images /255.
    print('max image:', np.max(images))
    print('min image:', np.min(images))
    return tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def create_generator():
    # """ Create Generator """
    generator = tf.keras.Sequential()

    # Input block
    generator.add(layers.Dense(7 * 7 * 128, use_bias=False, input_shape=(NOISE_DIMENSION,),
                               kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    # Reshape 1D Tensor into 3D
    generator.add(layers.Reshape((7, 7, 128)))
    # First upsampling block
    generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False,
                                         kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    # Second upsampling block
    generator.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                         kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    # Third upsampling block: note tanh, specific for DCGAN
    generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid',
                                         kernel_initializer=weight_init))
    # # Return generator

    return generator


def generate_noise(number_of_images=1, noise_dimension=NOISE_DIMENSION):
    """ Generate noise for number_of_images images, with a specific noise_dimension """
    return tf.random.normal([number_of_images, noise_dimension], mean=0.0, stddev=5.0)
    # return tf.random.uniform([number_of_images, noise_dimension], minval=-1., maxval=1.)


def create_discriminator():
    # """ Create Discriminator """
    discriminator = tf.keras.Sequential()
    # First Convolutional block
    discriminator.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                                    input_shape=[img_size, img_size, num_c], kernel_initializer=weight_init))
    discriminator.add(layers.LeakyReLU())
    discriminator.add(layers.Dropout(0.5))
    # Second Convolutional block
    discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=weight_init))
    discriminator.add(layers.LeakyReLU())
    discriminator.add(layers.Dropout(0.5))
    # Flatten and generate output prediction
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(1, kernel_initializer=weight_init, activation='sigmoid'))

    # Return discriminator
    return discriminator


def mix_pre(G_losses):
    used_l = tf.nn.softplus(lam)

    weights = tf.exp(used_l * G_losses)
    denom = tf.reduce_sum(weights)
    weights = tf.math.divide(weights, denom)
    print('lam_weights shape', weights.shape)
    g_loss = tf.reduce_sum(weights * G_losses)-0.001*used_l
    return g_loss, used_l


def compute_generator_loss(predicted_fake):
    """ Compute cross entropy loss for the generator """
    # G_losses = [cross_entropy_loss(tf.ones_like(predicted_fake[ind]), predicted_fake[ind]) for ind in
    #             range(NUM_OF_D)]
    # print('G_losses:', len(G_losses))
    G_losses = [tf.reduce_mean(tf.math.log(1 - predicted_fake[ind])) for ind in range(NUM_OF_D)]
    print('G_losses:', len(G_losses))
    G_losses, used_l = mix_pre(G_losses)
    return G_losses, used_l


def compute_discriminator_loss(predicted_real, predicted_fake):
    """ Compute discriminator loss """
    # loss_on_reals = [cross_entropy_loss(tensorflow.ones_like(predicted_real[ind]), predicted_real[ind]) for ind in
    #                  range(NUM_OF_D)]
    # loss_on_fakes = [cross_entropy_loss(tensorflow.zeros_like(predicted_fake[ind]), predicted_fake[ind]) for ind in
    #                  range(NUM_OF_D)]
    # D_losses = tensorflow.add(loss_on_reals, loss_on_fakes)
    # print('D_losses:', D_losses.shape)
    D_losses = [
        tf.reduce_mean(-tf.math.log(predicted_real[ind]) - tf.math.log(1 - predicted_fake[ind]))
        for ind in range(NUM_OF_D)]
    print('D_losses:', len(D_losses))

    return D_losses


def save_models(generator, discriminator, epoch):
    """ Save models at specific point in time. """
    tf.keras.models.save_model(
        generator,
        f'./{UNIQUE_RUN_ID}/generator_{epoch}.model',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    for ind in range(NUM_OF_D):
        tf.keras.models.save_model(
            discriminator[ind],
            f'./{UNIQUE_RUN_ID}/discriminator_{ind}_{epoch}.model',
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )


def print_training_progress(batch, generator_loss, discriminator_loss, used_l):
    """ Print training progress. """
    print('Losses after mini-batch %5d: used_l %e generator %e, mean_discriminator %e' %
          (batch, used_l, generator_loss, tf.reduce_mean(discriminator_loss)))


@tf.function
def perform_train_step(real_images, generator, discriminator):
    """ Perform one training step with Gradient Tapes """
    # Generate noise
    noise = generate_noise(BATCH_SIZE * NUM_OF_D)
    # Feed forward and loss computation for one batch
    with tf.GradientTape(persistent=True) as discriminator_tape, \
            tf.GradientTape(persistent=True) as generator_tape:
        # Generate images
        generated_images = generator(noise, training=True)
        print('g_images:', generated_images.shape)
        generated_images = tf.split(generated_images, NUM_OF_D)
        print('split_g_image:', len(generated_images))
        real_images = [tf.random.shuffle(real_images) for _ in range(NUM_OF_D)]
        print('r_image:', real_images[0].shape)
        print('split_r_image:', len(real_images))
        # Discriminate generated and real images
        discriminated_generated_images = [discriminator[ind](generated_images[ind], training=True) for ind in
                                          range(NUM_OF_D)]
        discriminated_real_images = [discriminator[ind](real_images[ind], training=True) for ind in range(NUM_OF_D)]
        # Compute loss
        generator_loss, used_l = compute_generator_loss(discriminated_generated_images)

        discriminator_loss = compute_discriminator_loss(discriminated_real_images, discriminated_generated_images)
        discriminator_loss = tf.split(discriminator_loss, NUM_OF_D)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(discriminated_real_images), tf.zeros_like(discriminated_generated_images)], axis=0),
                          tf.concat([discriminated_real_images, discriminated_generated_images], axis=0))

    # Compute gradients
    generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables+lam)

    discriminator_gradients = [
        discriminator_tape.gradient(discriminator_loss[ind], discriminator[ind].trainable_variables) for ind in
        range(NUM_OF_D)]
    del discriminator_tape, generator_tape
    # Optimize model using gradients

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables+lam))
    for ind in range(NUM_OF_D):
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients[ind], discriminator[ind].trainable_variables))

    # Return generator and discriminator losses
    return generator_loss, discriminator_loss, used_l,dc_acc


def train_gan(num_epochs, image_data, generator, discriminator):
    """ Train the GMAN """
    # tensorboard writer

    if not os.path.exists(f'./cruves/{UNIQUE_RUN_ID}'):
        os.mkdir(f'./cruves/{UNIQUE_RUN_ID}')
    writer = tf.summary.create_file_writer(f'./cruves/{UNIQUE_RUN_ID}/')
    with writer.as_default():
    # Perform one training step per batch for every epoch
        for epoch_no in range(num_epochs):
            num_batches = image_data.__len__()
            print(f'Starting epoch {epoch_no + 1} with {num_batches} batches...')
            batch_no = 0
            # Iterate over batches within epoch
            for batch in image_data:
                generator_loss, discriminator_loss, used_l,dc_acc = perform_train_step(batch, generator, discriminator)
                batch_no += 1
                # Print statistics and generate image after every n-th batch
                if batch_no % PRINT_STATS_AFTER_BATCH == 0:
                    print_training_progress(batch_no, generator_loss, discriminator_loss, used_l)

                    tf.summary.scalar("used_l", np.mean(used_l), epoch_no * ITERATIONS + batch_no)
                    tf.summary.scalar("gen_loss", np.mean(generator_loss), epoch_no * ITERATIONS + batch_no)
                    tf.summary.scalar("dc_acc", np.mean(dc_acc), epoch_no * ITERATIONS + batch_no)

                    for ind in range(NUM_OF_D):
                        tf.summary.scalar("dis[%d]_loss" % ind, np.mean(discriminator_loss[ind]),
                                          epoch_no * ITERATIONS + batch_no)


            generate_image(generator, epoch_no)

    # Finished :-)
    save_models(generator, discriminator, num_epochs)

    print(f'Finished unique run {UNIQUE_RUN_ID}')


def run_gan():
    """ Initialization and training """

    # Set random seed
    tf.random.set_seed(42)
    # Get image data
    data = load_data()
    # Create generator and discriminator
    generator = create_generator()

    discriminator = [create_discriminator() for _ in range(NUM_OF_D)]
    print('num of dis:', len(discriminator))
    # Train the GMAN
    print('Training GMAN ...')
    train_gan(NUM_EPOCHS, data, generator, discriminator)


if __name__ == '__main__':
    run_gan()
