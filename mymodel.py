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

img_width = 640
img_height = 480


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def build():
    print('Building model...')
    inputs = Input(shape=(img_height, img_width, 3))

    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv1)
    drop1 = Dropout(0.5)(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    avgpool1 = AveragePooling2D(pool_size=(2, 2))(drop1)
    add1 = add([maxpool1, avgpool1])
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(add1)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv2)
    drop2 = Dropout(0.5)(conv2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    avgpool2 = AveragePooling2D(pool_size=(2, 2))(drop2)
    add2 = add([maxpool2, avgpool2])
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(add2)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv3)
    drop3 = Dropout(0.5)(conv3)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    avgpool3 = AveragePooling2D(pool_size=(2, 2))(drop3)
    add3 = add([maxpool3, avgpool3])
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(add3)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    maxpool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    avgpool4 = AveragePooling2D(pool_size=(2, 2))(drop4)
    add4 = add([maxpool4, avgpool4])
    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same')(add4)
    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    maxpool5 = MaxPooling2D(pool_size=(2, 2))(drop5)
    avgpool5 = AveragePooling2D(pool_size=(2, 2))(drop5)
    add5 = add([maxpool5, avgpool5])

    conv6 = Conv2D(1024, (3, 3), activation='elu', padding='same')(add5)
    conv6 = Conv2D(1024, (3, 3), activation='elu', padding='same')(conv6)
    drop6 = Dropout(0.5)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(drop6), conv5], axis=3)
    conv7 = Conv2D(512, (3, 3), activation='elu', padding='same')(up7)
    conv7 = Conv2D(512, (3, 3), activation='elu', padding='same')(conv7)
    drop7 = Dropout(0.5)(conv7)
    up8 = concatenate([UpSampling2D(size=(2, 2))(drop7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='elu', padding='same')(up8)
    conv8 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv8)
    drop8 = Dropout(0.5)(conv8)
    up9 = concatenate([UpSampling2D(size=(2, 2))(drop8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='elu', padding='same')(up9)
    conv9 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv9)
    drop9 = Dropout(0.5)(conv9)
    up10 = concatenate([UpSampling2D(size=(2, 2))(drop9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='elu', padding='same')(up10)
    conv10 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv10)
    drop10 = Dropout(0.5)(conv10)
    up11 = concatenate([UpSampling2D(size=(2, 2))(drop10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='elu', padding='same')(up11)
    conv11 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv11)
    drop11 = Dropout(0)(conv11)

    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(drop11)

    model = Model(inputs=[inputs], outputs=[conv12])

    model.compile(optimizer=Adam(lr=L_0, decay=0.00001), loss=dice_coef_loss, metrics=[dice_coef])
    print('Model ready!')
    return model
