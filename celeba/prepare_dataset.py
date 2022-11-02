import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pathlib

# os.makedirs("celeba",exist_ok=True)
# url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
# output = "celeba/data.zip"
# gdown.download(url, output, quiet=True)
# with ZipFile("celeba/data.zip", "r") as zipobj:
#     zipobj.extractall("celeba")

# def preprocessing_celeba():
#     x_train, x_test = keras.preprocessing.image_dataset_from_directory(
#         "celeba",
#         label_mode=None,
#         image_size=(178, 178),
#         batch_size=100,
#         seed=42,
#         shuffle=True,
#         validation_split=1 / 18,
#         subset='both',
#     )
#     return x_train, x_test
#

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=train_buf)
# train_dataset = train_dataset.batch(batch_size)
# dataset = dataset.map(lambda x: x / 255.0)
# x_train, x_test = preprocessing_celeba()
# x_train = x_train.map(lambda x: x / 255.0)
# for x in x_train:
#     plt.axis("off")
#     plt.imshow((x.numpy() * 255).astype("int32")[0])
#     plt.savefig('sample_178.png')
#     break
# import pathlib
# import random
# from sklearn.model_selection import train_test_split
#
PROJECT_ROOT = pathlib.Path.cwd()
output_dir = PROJECT_ROOT / 'celeba'
output_dir.mkdir(exist_ok=True)
train_data=output_dir/'img_align_celeba'

valid_data = output_dir / 'valid_data'
valid_data.mkdir(exist_ok=True)

test_data = output_dir / 'test_data'
test_data.mkdir(exist_ok=True)

data_path = pathlib.Path('./celeba/')
#
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]  # 所有图片路径的列表

# random.shuffle(trainDir)
import shutil
trainDir = os.listdir(train_data)
testDir = os.listdir(test_data)
validDir = os.listdir(valid_data)
# if len(testDir) == 0:
#     for filename in all_image_paths[:10000]:
#         shutil.move(filename, test_data)
#
# if len(validDir) == 0:
#     for filename in all_image_paths [:1000]:
#         shutil.move(filename, valid_data)


print(len(testDir))
print(len(validDir))
print(len(trainDir))
