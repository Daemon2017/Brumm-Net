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


def block(input, size, drop):
    conv = Conv2D(size, (3, 3), activation='elu', padding='same')(input)
    conv = Conv2D(size, (3, 3), activation='elu', padding='same')(conv)
    conv = Dropout(drop)(conv)
    return conv


def build():
    print('Building model...')
    inputs = Input(shape=(img_height, img_width, 3))

    drop1 = block(inputs, 32, 0.0625)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    avgpool1 = AveragePooling2D(pool_size=(2, 2))(drop1)
    add1 = add([maxpool1, avgpool1])
    drop2 = block(add1, 64, 0.0625)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    avgpool2 = AveragePooling2D(pool_size=(2, 2))(drop2)
    add2 = add([maxpool2, avgpool2])
    drop3 = block(add2, 128, 0.0625)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    avgpool3 = AveragePooling2D(pool_size=(2, 2))(drop3)
    add3 = add([maxpool3, avgpool3])
    drop4 = block(add3, 256, 0.0625)
    maxpool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    avgpool4 = AveragePooling2D(pool_size=(2, 2))(drop4)
    add4 = add([maxpool4, avgpool4])
    drop5 = block(add4, 512, 0.0625)
    maxpool5 = MaxPooling2D(pool_size=(2, 2))(drop5)
    avgpool5 = AveragePooling2D(pool_size=(2, 2))(drop5)
    add5 = add([maxpool5, avgpool5])

    drop6 = block(add5, 1024, 0.0625)

    up7 = concatenate([UpSampling2D(size=(2, 2))(drop6), drop5], axis=3)
    drop7 = block(up7, 512, 0.0625)
    up8 = concatenate([UpSampling2D(size=(2, 2))(drop7), drop4], axis=3)
    drop8 = block(up8, 256, 0.0625)
    up9 = concatenate([UpSampling2D(size=(2, 2))(drop8), drop3], axis=3)
    drop9 = block(up9, 128, 0.0625)
    up10 = concatenate([UpSampling2D(size=(2, 2))(drop9), drop2], axis=3)
    drop10 = block(up10, 64, 0.0625)
    up11 = concatenate([UpSampling2D(size=(2, 2))(drop10), drop1], axis=3)
    drop11 = block(up11, 32, 0)

    conv12 = Conv2D(classes, (1, 1), activation='sigmoid')(drop11)

    model = Model(inputs=[inputs], outputs=[conv12])

    model.compile(optimizer=Adam(lr=L_0, decay=0.0001 / epochs_num), loss=dice_coef_loss, metrics=[dice_coef])
    print('Model ready!')
    return model
