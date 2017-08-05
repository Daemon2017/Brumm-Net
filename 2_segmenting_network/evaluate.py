import numpy as np
import os
import scipy.misc

from skimage.io import imread

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras import backend as K

K.set_image_dim_ordering('tf')

img_width = 1200
img_height = 800


def prepare_predict():
    print('Preparing prediction set...')
    files = os.listdir('./predict_raws/')
    x_files_names = filter(lambda x: x.endswith('_raw.jpg'), files)
    total = len(x_files_names)

    x_train = np.ndarray((total, img_height, img_width, 3), dtype=np.uint8)
    i = 0
    for x_file_name in x_files_names:
        img = imread(os.path.join('./predict_raws/' + x_file_name))
        x_train[i] = np.array([img])
        i += 1
    np.save('x_predict.npy', x_train)
    print('Prediction set prepared!')


def predict():
    x_predict = np.load('x_predict.npy')
    x_predict = x_predict.astype('float32')
    x_predict /= 255

    predictions = model.predict_on_batch(x_predict)
    np.save('predictions.npy', predictions)


def draw_predict():
    predictions = np.load('predictions.npy')
    i = 0
    for predict in predictions:
        scipy.misc.imsave('./predict_masks/' + str(i) + '.jpg', predict)
        i += 1


if not os.path.exists('predict_raws'):
    os.makedirs('predict_raws')

if not os.path.exists('predict_masks'):
    os.makedirs('predict_masks')

model.load_weights('model.h5')

scnd_choice = raw_input('Prepare predict data? (y or n): ')
if scnd_choice == 'y':
    prepare_predict()

thrd_choice = raw_input('Start prediction? (y or n): ')
if thrd_choice == 'y':
    predict()

frth_choice = raw_input('Save prediction to file? (y or n): ')
if frth_choice == 'y':
    draw_predict()
elif frth_choice == 'n':
    exit()
