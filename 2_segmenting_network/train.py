import numpy as np
import os
import scipy.misc

from skimage.io import imread

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras import backend as K

K.set_image_dim_ordering('tf')

tbCallBack = TensorBoard(log_dir='./logs',
                         histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)

img_width = 1200
img_height = 800


class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            self.model.save_weights('weights_batch.h5')
        self.batch += 1


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

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    drop1 = Dropout(0.05)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    drop2 = Dropout(0.05)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    drop3 = Dropout(0.05)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.05)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.05)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    drop6 = Dropout(0.05)(conv6)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(drop6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    drop7 = Dropout(0.05)(conv7)
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(drop7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    drop8 = Dropout(0.05)(conv8)
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(drop8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    drop9 = Dropout(0)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(drop9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])
    print('Model ready!')
    return model


def prepare_train():
    print('Preparing training set...')
    x_files = os.listdir('./raws/')
    x_files_names = filter(lambda x: x.endswith('_raw.jpg'), x_files)
    x_total = len(x_files_names)

    x_train = np.ndarray((x_total, img_height, img_width, 3), dtype=np.uint8)
    i = 0
    for x_file_name in x_files_names:
        x_img = imread(os.path.join('./raws/' + x_file_name))
        x_train[i] = np.array([x_img])
        i += 1
    np.save('x_train.npy', x_train)

    y_files = os.listdir('./masks/')
    y_files_names = filter(lambda x: x.endswith('_mask.jpg'), y_files)
    y_total = len(y_files_names)

    y_train = np.ndarray((y_total, img_height, img_width, 1), dtype=np.uint8)
    i = 0
    for y_file_name in y_files_names:
        y_img = scipy.ndimage.imread(os.path.join('./masks/' + y_file_name), mode='L')
        y_train[i] = np.array([y_img]).reshape(800, 1200, 1)
        i += 1
    np.save('y_train.npy', y_train)
    print('Training set prepared!')


def train():
    x_train = np.load('x_train.npy')
    x_train = x_train.astype('float32')
    x_mean = np.mean(x_train)
    x_std = np.std(x_train)
    x_train -= x_mean
    x_train /= x_std

    y_train = np.load('y_train.npy')
    y_train = y_train.astype('float32')
    y_train /= 255.

    model_checkpoint = ModelCheckpoint('weights_checkpoint.h5',
                                       monitor='val_loss',
                                       save_best_only=True)

    model.fit(x_train,
              y_train,
              batch_size=10,
              epochs=20,
              verbose=1,
              shuffle=True,
              validation_split=0.2,
              callbacks=[tbCallBack, model_checkpoint, WeightsSaver(model, 1)])
    model.save('model.h5')


if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('raws'):
    os.makedirs('raws')

if not os.path.exists('masks'):
    os.makedirs('masks')

zero_choice = raw_input('Prepare training data? (y or n): ')
if zero_choice == 'y':
    prepare_train()

frst_choice = raw_input('Start training? (y or n): ')
if frst_choice == 'y':
    model = build()
    train()
