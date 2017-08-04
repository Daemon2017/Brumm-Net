import cv2
import os


def prepare_mask(file_name):
    print('Preparing mask! Working with file' + str(file_name))

    img = cv2.imread('./raws/' + file_name)

    short_name = os.path.splitext(file_name)[0]
    cv2.imwrite('./raws/' + short_name + '_raw.jpg', img)
    os.remove('./raws/' + file_name)

    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./masks/' + short_name + '_mask.jpg', mask)


def enlarge_mask(file_name):
    print('Enlarging mask! Working with file' + str(file_name))
    im = cv2.imread('./masks/' + file_name)
    WHITE = [255, 255, 255]
    im = cv2.copyMakeBorder(im, 0, 1067, 0, 0, cv2.BORDER_CONSTANT, value=WHITE)
    cv2.imwrite('./masks/' + file_name, im)


def enlarge_raw(file_name):
    print('Enlarging raw! Working with file' + str(file_name))
    im = cv2.imread('./raws/' + file_name)
    BLACK = [0, 0, 0]
    im = cv2.copyMakeBorder(im, 0, 1067, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.imwrite('./raws/' + file_name, im)


def resize_mask(file_name):
    print('Resizing mask! Working with file' + str(file_name))
    im = cv2.imread('./masks/' + file_name)
    im = cv2.resize(im, (1008, 1008))
    cv2.imwrite('./masks/' + file_name, im)


def resize_raw(file_name):
    print('Resizing raw! Working with file' + str(file_name))
    im = cv2.imread('./raws/' + file_name)
    im = cv2.resize(im, (1008, 1008))
    cv2.imwrite('./raws/' + file_name, im)


if not os.path.exists('masks'):
    os.makedirs('masks')

if not os.path.exists('raws'):
    os.makedirs('raws')

files = os.listdir('./raws/')
images = filter(lambda x: x.endswith('.jpg'), files)
for image in images:
    prepare_mask(image)

files = os.listdir('./masks/')
images = filter(lambda x: x.endswith('_mask.jpg'), files)
for image in images:
    enlarge_mask(image)
files = os.listdir('./raws/')
images = filter(lambda x: x.endswith('_raw.jpg'), files)
for image in images:
    enlarge_raw(image)

files = os.listdir('./masks/')
images = filter(lambda x: x.endswith('_mask.jpg'), files)
for image in images:
    resize_mask(image)
files = os.listdir('./raws/')
images = filter(lambda x: x.endswith('_raw.jpg'), files)
for image in images:
    resize_raw(image)
