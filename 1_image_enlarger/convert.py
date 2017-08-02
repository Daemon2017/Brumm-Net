import cv2
import os

files = os.listdir('./')

x_files_names = filter(lambda x: x.endswith('_mask.jpg'), files)
for x_file_name in x_files_names:
    im = cv2.imread(x_file_name)
    WHITE = [255, 255, 255]
    im = cv2.copyMakeBorder(im, 0, 1067, 0, 0, cv2.BORDER_CONSTANT, value=WHITE)
    cv2.imwrite(x_file_name, im)

x_files_names = filter(lambda x: x.endswith('_raw.jpg'), files)
for x_file_name in x_files_names:
    im = cv2.imread(x_file_name)
    BLACK = [0, 0, 0]
    im = cv2.copyMakeBorder(im, 0, 1067, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.imwrite(x_file_name, im)