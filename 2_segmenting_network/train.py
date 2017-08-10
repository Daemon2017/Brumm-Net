import numpy as np
import os
import scipy.misc

from skimage.io import imread

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose, Dropout, BatchNormalization, add, \
    AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras import backend as K

K.set_image_dim_ordering('tf')

tbCallBack = TensorBoard(log_dir='./logs',
                         histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)

img_width = 1216
img_height = 800

size_of_batch = 5

start = 0
end = size_of_batch


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

    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv1)
    drop1 = Dropout(0.05)(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    avgpool1 = AveragePooling2D(pool_size=(2, 2))(drop1)
    add1 = add([maxpool1, avgpool1])
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(add1)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv2)
    drop2 = Dropout(0.05)(conv2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    avgpool2 = AveragePooling2D(pool_size=(2, 2))(drop2)
    add2 = add([maxpool2, avgpool2])
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(add2)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv3)
    drop3 = Dropout(0.05)(conv3)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    avgpool3 = AveragePooling2D(pool_size=(2, 2))(drop3)
    add3 = add([maxpool3, avgpool3])
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(add3)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv4)
    drop4 = Dropout(0.05)(conv4)
    maxpool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    avgpool4 = AveragePooling2D(pool_size=(2, 2))(drop4)
    add4 = add([maxpool4, avgpool4])
    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same')(add4)
    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same')(conv5)
    drop5 = Dropout(0.05)(conv5)
    maxpool5 = MaxPooling2D(pool_size=(2, 2))(drop5)
    avgpool5 = AveragePooling2D(pool_size=(2, 2))(drop5)
    add5 = add([maxpool5, avgpool5])

    conv6 = Conv2D(1024, (3, 3), activation='elu', padding='same')(add5)
    conv6 = Conv2D(1024, (3, 3), activation='elu', padding='same')(conv6)
    drop6 = Dropout(0.05)(conv6)

    up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop6), conv5], axis=3)
    conv7 = Conv2D(512, (3, 3), activation='elu', padding='same')(up7)
    conv7 = Conv2D(512, (3, 3), activation='elu', padding='same')(conv7)
    drop7 = Dropout(0.05)(conv7)
    up8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='elu', padding='same')(up8)
    conv8 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv8)
    drop8 = Dropout(0.05)(conv8)
    up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(drop8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='elu', padding='same')(up9)
    conv9 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv9)
    drop9 = Dropout(0.05)(conv9)
    up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(drop9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='elu', padding='same')(up10)
    conv10 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv10)
    drop10 = Dropout(0.05)(conv10)
    up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(drop10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='elu', padding='same')(up11)
    conv11 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv11)
    drop11 = Dropout(0)(conv11)

    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(drop11)

    model = Model(inputs=[inputs], outputs=[conv12])

    model.compile(optimizer=Adam(lr=0.0001, decay=0.0), loss=dice_coef_loss, metrics=[dice_coef])
    print('Model ready!')
    return model


def batch_generator():
    global start, end, total, x_files_names, y_files_names

    batch_num = 0
    while True:
        print('------------------------------')
        print('Generating batch ' + str(batch_num))
        x_train = np.ndarray((size_of_batch, img_height, img_width, 3), dtype=np.uint8)
        y_train = np.ndarray((size_of_batch, img_height, img_width, 1), dtype=np.uint8)

        sample = 0
        for j in range(start, end):
            print('Preparing file: #' + str(sample) + ', raw name: ' + str(x_files_names[j]) + ', mask name: ' + str(
                y_files_names[j]))
            x_img = imread(os.path.join('./raws/' + x_files_names[j]))
            y_img = scipy.ndimage.imread(os.path.join('./masks/' + y_files_names[j]), mode='L')
            x_train[sample] = np.array([x_img])
            y_train[sample] = np.array([y_img]).reshape(800, 1216, 1)
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

        print('Batch ' + str(batch_num) + ' generated!')
        batch_num += 1
        if batch_num == size_of_batch:
            batch_num = 0
        print('------------------------------')

        yield x_train, y_train


def train():
    print('Training...')
    model_checkpoint = ModelCheckpoint('weights_checkpoint.h5',
                                       monitor='val_loss',
                                       save_best_only=True)
    model.fit_generator(generator=batch_generator(),
                        epochs=25,
                        steps_per_epoch=total / size_of_batch,
                        verbose=1,
                        initial_epoch=0,
                        callbacks=[tbCallBack, model_checkpoint, WeightsSaver(model, 1)])
    model.save('model.h5')
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

print('------------------------------')
print('Raw files:')
for file in x_files_names:
    print(str(file))
print('------------------------------')
print('------------------------------')
print('Mask files:')
for file in y_files_names:
    print(str(file))
print('------------------------------')

total = 0
if x_total != y_total:
    exit()
else:
    total = x_total
    print('Number of X and Y in train is the same. Work continues!')

frst_choice = raw_input('Start training? (y or n): ')
if frst_choice == 'y':
    model = build()
    train()