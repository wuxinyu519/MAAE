import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
img_size=28
num_c=1
# n_labels=10
mse = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.BinaryCrossentropy()
def autoencoder_loss(inputs, reconstruction):
    return mse(inputs, reconstruction)
def reparameterization(z_mean, z_std):
    # Probabilistic with Gaussian posterior distribution
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0., stddev=5.)
    z = z_mean + tf.exp(z_std * .5) * epsilon
    return z

def ae_loss(encoder, decoder, images):
    z_mean,z_std=encoder(images, training=False)
    # encoder_output = encoder(images, training=False)
    # label = np.random.randint(0, high=n_labels, size=x_test_maam.shape[0])
    # labels = tf.reshape(tf.one_hot(label, n_labels), [x_test_maam.shape[0], 1, 1, n_labels])
    # decoder_output = decoder(tf.concat([encoder_output, labels], axis=-1), training=False)
    z=reparameterization(z_mean,z_std)
    decoder_output = decoder(z, training=False)
    ae_loss = autoencoder_loss(images, decoder_output)
    return ae_loss

def ad_loss(discriminator, encoder, images):
    z_mean, z_std = encoder(images, training=False)
    z = reparameterization(z_mean, z_std)
    print(len(discriminator))
    if len(discriminator)>1:
        fake = [discriminator[ind](z, training=False) for ind in
                   range(len(discriminator))]
        G_losses = [cross_entropy(tf.ones_like(fake[ind]), fake[ind]) for ind in range(len(discriminator))]
        return tf.reduce_mean(G_losses)
    else:
        fake=discriminator[0](z, training=False)
        return cross_entropy(tf.ones_like(fake), fake)

if __name__ == '__main__':
    (_, _), (x_test_maam, _) = tf.keras.datasets.mnist.load_data()
    x_test_maam = x_test_maam.astype('float32') / 255.
    x_test_maam = x_test_maam.reshape(x_test_maam.shape[0], img_size*img_size*num_c)



    # model_dec_1 = tf.keras.models.load_model("./unsupervised_aae_gaussian_posterior_dense_8z_testlr/decoder_199.model/")
    # model_enc_1 = tf.keras.models.load_model("./unsupervised_aae_gaussian_posterior_dense_8z_testlr/encoder_199.model/")
    # model_dis=[tf.keras.models.load_model("./unsupervised_aae_gaussian_posterior_dense_8z_testlr/discriminator_199"
    #                                        ".model/")]
    model_dis = []
    model_dec_1 = tf.keras.models.load_model(
        "./unsupervised_3d_maae_mean_gaussian_posterior_dense_8z_testlr/decoder_199.model/")
    model_enc_1 = tf.keras.models.load_model(
        "./unsupervised_3d_maae_mean_gaussian_posterior_dense_8z_testlr/encoder_199.model/")
    model_dis_1 = tf.keras.models.load_model(
        "./unsupervised_3d_maae_mean_gaussian_posterior_dense_8z_testlr/discriminator0_199"
        ".model/")
    model_dis_2 = tf.keras.models.load_model(
        "./unsupervised_3d_maae_mean_gaussian_posterior_dense_8z_testlr/discriminator1_199"
        ".model/")
    model_dis_3 = tf.keras.models.load_model(
        "./unsupervised_3d_maae_mean_gaussian_posterior_dense_8z_testlr/discriminator2_199"
        ".model/")
    model_dis.append(model_dis_1)
    model_dis.append(model_dis_2)
    model_dis.append(model_dis_3)
    # model_dec_1 = tf.keras.models.load_model(
    #     "./unsupervised_3d_maae_mean_gaussian_posterior_dense_2z_testlr/decoder_199.model/")
    # model_enc_1 = tf.keras.models.load_model(
    #     "./unsupervised_3d_maae_mean_gaussian_posterior_dense_2z_testlr/encoder_199.model/")
    model_3dis = []
    model_dec_2 = tf.keras.models.load_model("./unsupervised_3d_maae_max_gaussian_posterior_dense_8z_test/decoder_199.model/")
    model_enc_2 = tf.keras.models.load_model("./unsupervised_3d_maae_max_gaussian_posterior_dense_8z_test/encoder_199.model/")
    model_dis_1 = tf.keras.models.load_model("./unsupervised_3d_maae_max_gaussian_posterior_dense_8z_test/discriminator0_199"
                                             ".model/")
    model_dis_2 = tf.keras.models.load_model("./unsupervised_3d_maae_max_gaussian_posterior_dense_8z_test/discriminator1_199"
                                             ".model/")
    model_dis_3 = tf.keras.models.load_model("./unsupervised_3d_maae_max_gaussian_posterior_dense_8z_test/discriminator2_199"
                                             ".model/")
    model_3dis.append(model_dis_1)
    model_3dis.append(model_dis_2)
    model_3dis.append(model_dis_3)
    # model2 vs model1(>0,model2>model1;<0,model2<model1)
    # b_enc_a_dec=ae_loss(model_enc_2,model_dec_1,x_test_maam)
    # a_enc_a_dec=ae_loss(model_enc_1,model_dec_1,x_test_maam)
    #
    # a_enc_b_dec=ae_loss(model_enc_1,model_dec_2,x_test_maam)
    # b_enc_b_dec=ae_loss(model_enc_2,model_dec_2,x_test_maam)

    a_enc_b_dec=ae_loss(model_enc_1,model_dec_2,x_test_maam)+ad_loss(model_3dis,model_enc_1,x_test_maam)
    a_enc_a_dec = ae_loss(model_enc_1, model_dec_1, x_test_maam)+ad_loss(model_dis,model_enc_1,x_test_maam)
    b_enc_a_dec = ae_loss(model_enc_2, model_dec_1, x_test_maam)+ad_loss(model_dis,model_enc_2,x_test_maam)
    b_enc_b_dec = ae_loss(model_enc_2, model_dec_2, x_test_maam)+ad_loss(model_3dis,model_enc_2,x_test_maam)

    print('AMAM SCORE:%.4f'%-np.log((b_enc_a_dec/a_enc_a_dec) / (a_enc_b_dec / b_enc_b_dec)))
