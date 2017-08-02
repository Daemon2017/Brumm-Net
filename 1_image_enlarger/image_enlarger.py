import cv2
import os

if not os.path.exists('masks'):
    os.makedirs('masks')

if not os.path.exists('raws'):
    os.makedirs('raws')

files = os.listdir('./masks/')
x_files_names = filter(lambda x: x.endswith('_mask.jpg'), files)
for x_file_name in x_files_names:
    im = cv2.imread('./masks/' + x_file_name)
    WHITE = [255, 255, 255]
    im = cv2.copyMakeBorder(im, 0, 1067, 0, 0, cv2.BORDER_CONSTANT, value=WHITE)
    cv2.imwrite('./masks/' + x_file_name, im)

files = os.listdir('./raws/')
x_files_names = filter(lambda x: x.endswith('_raw.jpg'), files)
for x_file_name in x_files_names:
    im = cv2.imread('./raws/' + x_file_name)
    BLACK = [0, 0, 0]
    im = cv2.copyMakeBorder(im, 0, 1067, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.imwrite('./raws/' + x_file_name, im)
