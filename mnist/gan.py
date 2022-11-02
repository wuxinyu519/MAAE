# Import
import time

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
# Initialize variables
NUM_EPOCHS = 25
img_size=28
num_c=1
BUFFER_SIZE = 59000
BATCH_SIZE = 100
ITERATIONS = int(BUFFER_SIZE / BATCH_SIZE)
NOISE_DIMENSION = 75
UNIQUE_RUN_ID = str('gan_75z')
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

experiment_dir = output_dir / 'gan_75z'
experiment_dir.mkdir(exist_ok=True)

fake_dir = experiment_dir / 'fake_img'
fake_dir.mkdir(exist_ok=True)



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
    images, x_valid = train_test_split(
        images,  # 把上面剩余的 x_train, y_train继续拿来切
        test_size=1 / 60  # test_size默认是0.25
    )
    images = images.astype('float32')

    images = images / 255.
    images = images.reshape(images.shape[0], img_size, img_size, num_c)

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

    generator.add(layers.Reshape((7, 7, 128)))

    generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False,
                                         kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())

    generator.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                         kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())

    generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid',
                                         kernel_initializer=weight_init))
    # # Return generator

    return generator


def generate_noise(number_of_images=1, noise_dimension=NOISE_DIMENSION):
    """ Generate noise for number_of_images images, with a specific noise_dimension """
    return tf.random.normal([number_of_images, noise_dimension], mean=0.0, stddev=5.0)


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



def compute_generator_loss(predicted_fake):
    """ Compute cross entropy loss for the generator """
    G_losses = tf.reduce_mean(tf.math.log(1 - predicted_fake))
    return G_losses


def compute_discriminator_loss(predicted_real, predicted_fake):
    """ Compute discriminator loss """
    D_losses =tf.reduce_mean(-tf.math.log(predicted_real) - tf.math.log(1 - predicted_fake))



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
    tf.keras.models.save_model(
        discriminator,
        f'./{UNIQUE_RUN_ID}/discriminator_{epoch}.model',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
        )


def print_training_progress(batch, generator_loss, discriminator_loss):
    """ Print training progress. """
    print('Losses after mini-batch %5d:  generator %f, mean_discriminator %f' %
          (batch, generator_loss, tf.reduce_mean(discriminator_loss)))


@tf.function
def perform_train_step(real_images, generator, discriminator):
    """ Perform one training step with Gradient Tapes """
    # Generate noise
    noise = generate_noise(BATCH_SIZE)
    # Feed forward and loss computation for one batch
    with tf.GradientTape(persistent=True) as discriminator_tape, \
            tf.GradientTape(persistent=True) as generator_tape:
        # Generate images
        generated_images = generator(noise, training=True)

        # Discriminate generated and real images
        discriminated_generated_images = discriminator(generated_images, training=True)
        discriminated_real_images = discriminator(real_images, training=True)
        # Compute loss
        generator_loss= compute_generator_loss(discriminated_generated_images)

        discriminator_loss = compute_discriminator_loss(discriminated_real_images, discriminated_generated_images)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(discriminated_real_images), tf.zeros_like(discriminated_generated_images)], axis=0),
                          tf.concat([discriminated_real_images, discriminated_generated_images], axis=0))

    # Compute gradients
    generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables+lam)

    discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)

    del discriminator_tape, generator_tape
    # Optimize model using gradients

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables+lam))
    discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables))

    # Return generator and discriminator losses
    return generator_loss, discriminator_loss,dc_acc


def train_gan(num_epochs, image_data, generator, discriminator):
    """ Train the GAN """
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
                generator_loss, discriminator_loss,dc_acc = perform_train_step(batch, generator, discriminator)
                batch_no += 1
                # Print statistics and generate image after every n-th batch
                if batch_no % PRINT_STATS_AFTER_BATCH == 0:
                    print_training_progress(batch_no, generator_loss, discriminator_loss)
                    tf.summary.scalar("gen_loss", np.mean(generator_loss), epoch_no * ITERATIONS + batch_no)
                    tf.summary.scalar("dc_acc", np.mean(dc_acc), epoch_no * ITERATIONS + batch_no)

                    tf.summary.scalar("dis_loss" , np.mean(discriminator_loss),
                                          epoch_no * ITERATIONS + batch_no)

            generate_image(generator, epoch_no)
            save_models(generator, discriminator, epoch_no)

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

    discriminator = create_discriminator()

    # Train the GMAN
    print('Training GAN ...')
    train_gan(NUM_EPOCHS, data, generator, discriminator)


if __name__ == '__main__':
    run_gan()
