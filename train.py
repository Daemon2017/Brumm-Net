import numpy as np
import os
import scipy.misc

from skimage.io import imread

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose, Dropout, BatchNormalization, add, \
    AveragePooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, LearningRateScheduler
from keras import backend as K

from mymodel import dice_coef, dice_coef_loss, build, img_width, img_height, classes, L_0, epochs_num

size_of_batch = 1

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


def train():
    print('Training...')
    data_gen_args_train = dict(featurewise_center=False,
                               samplewise_center=False,
                               featurewise_std_normalization=False,
                               samplewise_std_normalization=False,
                               zca_whitening=False,
                               zca_epsilon=1e-6,
                               rotation_range=90.,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.,
                               zoom_range=0.2,
                               channel_shift_range=0.,
                               fill_mode='constant',
                               cval=0.,
                               horizontal_flip=True,
                               vertical_flip=False,
                               rescale=1. / 255.,
                               preprocessing_function=None,
                               data_format=K.image_data_format())
    train_image_datagen = ImageDataGenerator(**data_gen_args_train)
    train_mask_datagen = ImageDataGenerator(**data_gen_args_train)
    seed = 1
    train_image_generator = train_image_datagen.flow_from_directory(directory='./raws',
                                                                    target_size=(img_height, img_width),
                                                                    color_mode='rgb',
                                                                    classes=None,
                                                                    class_mode=None,
                                                                    batch_size=size_of_batch,
                                                                    shuffle=False,
                                                                    seed=seed)
    train_mask_generator = train_mask_datagen.flow_from_directory(directory='./masks',
                                                                  target_size=(img_height, img_width),
                                                                  color_mode='grayscale',
                                                                  classes=None,
                                                                  class_mode=None,
                                                                  batch_size=size_of_batch,
                                                                  shuffle=False,
                                                                  seed=seed)
    train_generator = zip(train_image_generator, train_mask_generator)

    data_gen_args_test = dict(featurewise_center=False,
                              samplewise_center=False,
                              featurewise_std_normalization=False,
                              samplewise_std_normalization=False,
                              zca_whitening=False,
                              zca_epsilon=1e-6,
                              rotation_range=0.,
                              width_shift_range=0.,
                              height_shift_range=0.,
                              shear_range=0.,
                              zoom_range=0.,
                              channel_shift_range=0.,
                              fill_mode='constant',
                              cval=0.,
                              horizontal_flip=False,
                              vertical_flip=False,
                              rescale=1. / 255.,
                              preprocessing_function=None,
                              data_format=K.image_data_format())
    test_image_datagen = ImageDataGenerator(**data_gen_args_test)
    test_mask_datagen = ImageDataGenerator(**data_gen_args_test)
    seed = 1
    test_image_generator = test_image_datagen.flow_from_directory(directory='./test_raws',
                                                                  target_size=(img_height, img_width),
                                                                  color_mode='rgb',
                                                                  classes=None,
                                                                  class_mode=None,
                                                                  batch_size=size_of_batch,
                                                                  shuffle=False,
                                                                  seed=seed)
    test_mask_generator = test_mask_datagen.flow_from_directory(directory='./test_masks',
                                                                target_size=(img_height, img_width),
                                                                color_mode='grayscale',
                                                                classes=None,
                                                                class_mode=None,
                                                                batch_size=size_of_batch,
                                                                shuffle=False,
                                                                seed=seed)
    test_generator = zip(test_image_generator, test_mask_generator)

    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
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
