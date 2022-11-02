import numpy
import tensorflow as tf
import os
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def generate_fake_images(model, NUM):
    """ Generate subplots with generated examples. """
    z = tf.random.normal([NUM, z_dim], mean=0.0, stddev=5.0)
    # dec_input = np.reshape(z, (NUM, z_dim))

    # x = decoder(dec_input.astype('float32'), training=False)
    x = model(z, training=False)
    x = x
    x = numpy.reshape(x, (x_test.shape[0], img_size, img_size, num_c))
    # x = x * 255.
    x = (x.astype('float32') - 0.5) / 0.5
    return x


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    act1 = []
    act2 = []
    for batch, (batch_x1) in enumerate(images1):
        # resize images
        batch_x1 = scale_images(batch_x1, (299, 299, 3))
        # pre-process images
        batch_x1 = preprocess_input(batch_x1)

        # calculate activations
        act1.append(model.predict(batch_x1))

    act1 = numpy.concatenate(act1, axis=0)
    print(act1[0].shape)
    for batch, (batch_x2) in enumerate(images2):
        batch_x2 = scale_images(batch_x2, (299, 299, 3))
        batch_x2 = preprocess_input(batch_x2)
        # calculate activations
        act2.append(model.predict(batch_x2))
        # calculate mean and covariance statistics

    act2 = numpy.concatenate(act2, axis=0)
    print(len(act2))
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
decoder = tf.keras.models.load_model("./unsupervised_3d_maae_max_gaussian_posterior_dense_8z/decoder_199.model/")
# decoder = tf.keras.models.load_model("./supervised_aae_mixture_posterior_dense_2z/decoder_2.model/")

# generator = tf.keras.models.load_model("./gman_lam_75z/generator_24.model/")

img_size = 28
num_c = 1
z_dim = 8
batch_size = 100
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = (x_test.astype('float32')-127.5)/127.5
# x_test = x_test.astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], img_size, img_size, num_c)

# define two fake collections of images
images1 = generate_fake_images(decoder, x_test.shape[0])

images2 = x_test
print('Prepared', images1.shape, images2.shape)

# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# fid between images1 and images2
images1 = tf.data.Dataset.from_tensor_slices(images1).batch(batch_size)
images2 = tf.data.Dataset.from_tensor_slices(images2).batch(batch_size)
fid = calculate_fid(model, images1, images2)
print('FID (different): %.3f' % fid)
print("unsupervised_3d_maae_max_gaussian_posterior_dense_8z/decoder_199.mod")