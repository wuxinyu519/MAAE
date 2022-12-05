# MAAE

===============================

Multi-adversarial autoencoder appendix

Archittecture
----------------

![architecture](./maae_architecture_v4.png "MAAE")


Experiment Setup
----------------

We apply a convolution architecture on CelebA dataset, as shown below:

| Encoder      | output shape |     Decoder     |     output     |     |
|:-------------|:------------:|:---------------:|:--------------:|:---:|
| InputLayer   |  (32, 32, 3) |    InputLayer   |   (1, 1, 15)   |     |
| conv2d       | (16, 16, 32) |      dense      |  (1, 1, 8192)  |     |
| LeakyReLU    | (16, 16, 32) | conv2dtranspose | ( 16, 16, 128) |     |
| BatchNorm    | (16, 16, 32) |    BatchNorm    |  (16, 16, 128) |     |
| conv2d       |  ( 8, 8, 64) |    LeakyReLU    |  (16, 16, 128) |     |
| LeakyReLU    |  (8, 8, 64)  | conv2dtranspose |  (32, 32, 64)  |     |
| BatchNorm    |  (8, 8, 64)  |    BatchNorm    |  (32, 32, 64)  |     |
| conv2d       |  (4, 4, 128) |    LeakyReLU    |  (32, 32, 64)  |     |
| LeakyReLU    |  (4, 4, 128) | conv2dtranspose |   (32, 32, 3)  |     |
| BatchNorm    |  (4, 4, 128) |                 |                |     |
| flatten      |    (2048)    |                 |                |     |
| dense(mean)  |   (1,1,15)   |                 |                |     |
| dense(stddv) |   (1,1,15)   |                 |                |     |

