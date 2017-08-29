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

from mymodel import dice_coef, dice_coef_loss, build, img_width, img_height, classes, L_0, epochs_num

size_of_batch = 5

start = 0
end = size_of_batch

start_test = 0
end_test = size_of_batch

K.set_image_dim_ordering('tf')

tbCallBack = TensorBoard(log_dir='./logs',
                         histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)


class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            self.model.save_weights('weights_batch.h5')
        self.batch += 1


def batch_generator():
    global start, end, total, x_files_names, y_files_names

    batch_num = 0
    while True:
        print('------------------------------')
        print('Generating training batch ' + str(batch_num))
        x_train = np.ndarray((size_of_batch, img_height, img_width, 3), dtype=np.uint8)
        y_train = np.ndarray((size_of_batch, img_height, img_width, 1), dtype=np.uint8)

        sample = 0
        for j in range(start, end):
            print(
                'Preparing training file: #' + str(sample) + ', raw name: ' + str(
                    x_files_names[j]) + ', mask name: ' + str(
                    y_files_names[j]))
            x_img = imread(os.path.join('./raws/' + x_files_names[j]))
            y_img = scipy.ndimage.imread(os.path.join('./masks/' + y_files_names[j]), mode='L')
            x_train[sample] = np.array([x_img])
            y_train[sample] = np.array([y_img]).reshape(img_height, img_width, 1)
            sample += 1

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_mean = np.mean(x_train)
        x_std = np.std(x_train)
        x_train -= x_mean
        x_train /= x_std
        y_train /= 255.0

        print('Start is ' + str(start) + ', end is ' + str(end))
        start += size_of_batch
        end += size_of_batch
        if end > total:
            start = 0
            end = size_of_batch

        print('Training batch ' + str(batch_num) + ' generated!')
        batch_num += 1
        if batch_num == size_of_batch:
            batch_num = 0
        print('------------------------------')

        yield x_train, y_train


def batch_test_generator():
    global start_test, end_test, total_test, x_test_files_names, y_test_files_names

    batch_num = 0
    while True:
        print('------------------------------')
        print('Generating test batch ' + str(batch_num))
        x_test = np.ndarray((size_of_batch, img_height, img_width, 3), dtype=np.uint8)
        y_test = np.ndarray((size_of_batch, img_height, img_width, 1), dtype=np.uint8)

        sample = 0
        for j in range(start_test, end_test):
            print('Preparing test file: #' + str(sample) + ', raw name: ' + str(
                x_test_files_names[j]) + ', mask name: ' + str(
                y_test_files_names[j]))
            x_img = imread(os.path.join('./test_raws/' + x_test_files_names[j]))
            y_img = scipy.ndimage.imread(os.path.join('./test_masks/' + y_test_files_names[j]), mode='L')
            x_test[sample] = np.array([x_img])
            y_test[sample] = np.array([y_img]).reshape(img_height, img_width, 1)
            sample += 1

        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')
        x_test_mean = np.mean(x_test)
        x_test_std = np.std(x_test)
        x_test -= x_test_mean
        x_test /= x_test_std
        y_test /= 255.0

        print('Start is ' + str(start_test) + ', end is ' + str(end_test))
        start_test += size_of_batch
        end_test += size_of_batch
        if end_test > total_test:
            start_test = 0
            end_test = size_of_batch

        print('Test batch ' + str(batch_num) + ' generated!')
        batch_num += 1
        if batch_num == size_of_batch:
            batch_num = 0
        print('------------------------------')

        yield x_test, y_test


def train():
    print('Training...')
    model.fit_generator(generator=batch_generator(),
                        validation_data=batch_test_generator(),
                        epochs=epochs_num,
                        steps_per_epoch=total / size_of_batch,
                        validation_steps=total_test / size_of_batch,
                        verbose=1,
                        initial_epoch=0,
                        callbacks=[tbCallBack, WeightsSaver(model, 1)])
    print('Training ended!')


if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('raws'):
    os.makedirs('raws')
if not os.path.exists('masks'):
    os.makedirs('masks')
x_files = os.listdir('./raws/')
y_files = os.listdir('./masks/')
x_files_names = filter(lambda x: x.endswith('_raw.jpg'), x_files)
y_files_names = filter(lambda x: x.endswith('_mask.jpg'), y_files)
x_files_names.sort()
y_files_names.sort()
x_total = len(x_files_names)
y_total = len(y_files_names)
total = 0
if x_total != y_total:
    exit()
else:
    total = x_total
    print('Number of X and Y in train is the same. Work continues!')

if not os.path.exists('test_raws'):
    os.makedirs('test_raws')
if not os.path.exists('test_masks'):
    os.makedirs('test_masks')
x_test_files = os.listdir('./test_raws/')
y_test_files = os.listdir('./test_masks/')
x_test_files_names = filter(lambda x: x.endswith('_raw.jpg'), x_test_files)
y_test_files_names = filter(lambda x: x.endswith('_mask.jpg'), y_test_files)
x_test_files_names.sort()
y_test_files_names.sort()
x_test_total = len(x_test_files_names)
y_test_total = len(y_test_files_names)
total_test = 0
if x_test_total != y_test_total:
    exit()
else:
    total_test = x_test_total
    print('Number of X and Y in test is the same. Work continues!')

frst_choice = raw_input('Start training? (y or n): ')
if frst_choice == 'y':
    model = build()
    train()
