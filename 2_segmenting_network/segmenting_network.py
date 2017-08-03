import numpy as np
import os

from skimage.io import imread, imsave

from keras.models import load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K

K.set_image_dim_ordering('tf')

tbCallBack = TensorBoard(log_dir='./logs',
                         histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def build():
    inputs = Input(shape=(1008, 1008, 3))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def prepare_train():
    files = os.listdir('./raws/')
    x_files_names = filter(lambda x: x.endswith('_raw.jpg'), files)
    total = len(x_files_names)

    x_train = np.ndarray((total, 1008, 1008, 3), dtype=np.uint8)
    i = 0
    for x_file_name in x_files_names:
        img = imread(os.path.join('./raws/' + x_file_name))
        x_train[i] = np.array([img])
        i += 1
    np.save('x_train.npy', x_train)

    files = os.listdir('./masks/')
    y_files_names = filter(lambda x: x.endswith('_mask.jpg'), files)
    total = len(y_files_names)

    y_train = np.ndarray((total, 1008, 1008, 3), dtype=np.uint8)
    i = 0
    for y_file_name in y_files_names:
        img = imread(os.path.join('./masks/' + y_file_name))
        y_train[i] = np.array([img])
        i += 1
    np.save('y_train.npy', y_train)


def train():
    x_train = np.load('x_train.npy')
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np.load('y_train.npy')
    y_train = y_train.astype('float32')
    y_train /= 255.

    model.fit(x_train,
              y_train,
              batch_size=4,
              epochs=25,
              callbacks=[tbCallBack])
    model.save('model.h5')


def prepare_predict():
    files = os.listdir('./predict_raws/')
    x_files_names = filter(lambda x: x.endswith('_raw.jpg'), files)
    total = len(x_files_names)

    x_train = np.ndarray((total, 1008, 1008, 3), dtype=np.uint8)
    i = 0
    for x_file_name in x_files_names:
        img = imread(os.path.join('./predict_raws/' + x_file_name))
        x_train[i] = np.array([img])
        i += 1
    np.save('x_predict.npy', x_train)


def predict():
    x_predict = np.load('x_predict.npy')
    x_predict = x_predict.astype('float32')
    x_predict /= 255

    predictions = model.predict_on_batch(x_predict)
    np.save('predictions.npy', predictions)


if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('raws'):
    os.makedirs('raws')

if not os.path.exists('masks'):
    os.makedirs('masks')

if not os.path.exists('predict_raws'):
    os.makedirs('predict_raws')

if not os.path.exists('predict_masks'):
    os.makedirs('predict_masks')

zero_choice = raw_input('Prepare training data? (y or n): ')
if zero_choice == 'y':
    prepare_train()

frst_choice = raw_input('Please, enter needed action (load or train): ')
if frst_choice == 'load':
    model = load_model('model.h5')
elif frst_choice == 'train':
    model = build()
    train()

scnd_choice = raw_input('Prepare test data? (y or n): ')
if scnd_choice == 'y':
    prepare_predict()

thrd_choice = raw_input('Model is ready! Start prediction? (y or n): ')
if thrd_choice == 'y':
    predict()
elif thrd_choice == 'n':
    exit()
