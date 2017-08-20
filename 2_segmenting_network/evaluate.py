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

from mymodel import dice_coef, dice_coef_loss, build, img_width, img_height

def predict():
    print('Preparing prediction set...')
    files = os.listdir('./predict_raws/')
    x_files_names = filter(lambda x: x.endswith('_raw.jpg'), files)
    total = len(x_files_names)

    x_predict = np.ndarray((total, img_height, img_width, 3), dtype=np.uint8)
    i = 0
    for x_file_name in x_files_names:
        img = imread(os.path.join('./predict_raws/' + x_file_name))
        x_predict[i] = np.array([img])
        i += 1
    print('Prediction set prepared!')

    x_predict = x_predict.astype('float32')
    x_mean = np.mean(x_predict)
    x_std = np.std(x_predict)
    x_predict -= x_mean
    x_predict /= x_std

    predictions = model.predict_on_batch(x_predict)
    i = 0
    for prediction in predictions:
        prediction = (prediction[:, :, 0] * 255.).astype(np.uint8)
        short_name = os.path.splitext(x_files_names[i])[0]
        scipy.misc.imsave('./predict_masks/' + str(short_name) + '_mask.jpg', prediction)
        i += 1


if not os.path.exists('predict_raws'):
    os.makedirs('predict_raws')

if not os.path.exists('predict_masks'):
    os.makedirs('predict_masks')

model = build()
model.load_weights('weights_batch.h5')

thrd_choice = raw_input('Start prediction? (y or n): ')
if thrd_choice == 'y':
    predict()
