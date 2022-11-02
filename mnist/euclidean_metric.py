import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
img_size=28
num_c=1
n_labels=10
z_dim=75
NUM=5

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train =x_train.astype('float32') /255.
x_train = np.reshape(x_train, (x_train.shape[0], img_size*img_size*num_c))
decoder = tf.keras.models.load_model("./unsupervised_3d_maae_lam_gaussian_posterior_dense_75z_testlr/decoder_49.model/")
# generator = tf.keras.models.load_model("./gman_lam_8z/generator_25.model/")
def generate_fake_images(decoder, NUM):

    i=1

    fig = plt.figure(figsize=(NUM,2))

    for _ in range(NUM):
        dis = []
        """ Generate subplots with generated examples. """
        # z = tf.random.normal([1, z_dim], mean=0.0, stddev=5.0)
        z = tf.random.uniform([1, z_dim], minval=-1., maxval=1.)
        dec_input = np.reshape(z, (1, z_dim))
        x = decoder(dec_input.astype('float32'), training=False).numpy()
        axis=plt.subplot(NUM,2,i)
        plt.imshow(x.reshape(img_size, img_size))
        plt.gray()
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        x = np.array(x.reshape(img_size * img_size))
        for real_img in x_train:
            dis.append(eucliDist(x, real_img))
        # list_dis=dis.tolist()

        min_index = dis.index(min(dis))
        axis=plt.subplot(NUM, 2, i + 1)
        plt.imshow(x_train[min_index].reshape(img_size, img_size))
        plt.gray()
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

        i=i+2
        # fake.append(x)
    plt.savefig('nearest_img.png')
    plt.close('all')



def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))
    # return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

generate_fake_images(decoder, NUM)