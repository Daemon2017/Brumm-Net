import numpy as np
import os
import scipy.misc

from skimage.io import imread

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose, Dropout, BatchNormalization, add, \
    AveragePooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, LearningRateScheduler
from keras import backend as K

K.set_image_dim_ordering('tf')

L_0 = 0.0001

img_width = 480
img_height = 352
classes = 1

epochs_num = 10


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def block_a(input, size):
    conv1 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)

    conv2 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)
    conv2 = Conv2D(size, (3, 3), activation='elu', padding='same')(conv2)

    conv3 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)
    conv3 = Conv2D(size, (3, 3), activation='elu', padding='same')(conv3)
    conv3 = Conv2D(size, (3, 3), activation='elu', padding='same')(conv3)

    conv = concatenate([conv1, conv2, conv3], axis=3)
    return conv


def block_b(input, size):
    conv1 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)

    conv2 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)
    conv2 = Conv2D(size, (1, 7), activation='elu', padding='same')(conv2)
    conv2 = Conv2D(size, (7, 1), activation='elu', padding='same')(conv2)

    conv3 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)
    conv3 = Conv2D(size, (7, 1), activation='elu', padding='same')(conv3)
    conv3 = Conv2D(size, (1, 7), activation='elu', padding='same')(conv3)
    conv3 = Conv2D(size, (7, 1), activation='elu', padding='same')(conv3)
    conv3 = Conv2D(size, (1, 7), activation='elu', padding='same')(conv3)

    conv = concatenate([conv1, conv2, conv3], axis=3)
    return conv


def block_c(input, size):
    conv1 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)

    conv2 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)
    conv2_1 = Conv2D(size, (1, 3), activation='elu', padding='same')(conv2)
    conv2_2 = Conv2D(size, (3, 1), activation='elu', padding='same')(conv2)

    conv3 = Conv2D(size, (1, 1), activation='elu', padding='same')(input)
    conv3 = Conv2D(size, (3, 1), activation='elu', padding='same')(conv3)
    conv3 = Conv2D(size, (1, 3), activation='elu', padding='same')(conv3)
    conv3_1 = Conv2D(size, (1, 3), activation='elu', padding='same')(conv3)
    conv3_2 = Conv2D(size, (3, 1), activation='elu', padding='same')(conv3)

    conv = concatenate([conv1, conv2_1, conv2_2, conv3_1, conv3_2], axis=3)
    return conv


def build():
    print('Building model...')
    inputs = Input(shape=(img_height, img_width, 3))

    conv1 = block_a(inputs, 128)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = block_b(pool1, 256)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = block_c(pool2, 512)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(1024, (1, 1), activation='elu', padding='same')(pool3)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3)
    conv5 = block_c(up5, 512)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=3)
    conv6 = block_b(up6, 256)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)
    conv7 = block_a(up7, 128)

    conv8 = Conv2D(classes, (1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    model.compile(optimizer=Adam(lr=L_0, decay=0.0001 / epochs_num), loss=dice_coef_loss, metrics=[dice_coef])
    print('Model ready!')
    return model
