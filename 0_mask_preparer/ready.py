import cv2
import os

def prepare_mask(file_name):
    print('Working with file' + str(file_name))

    img = cv2.imread(file_name)
	
	short_name = os.path.splitext(file_name)[0]
	cv2.imwrite(short_name + '_raw.jpg', img)
	os.remove(file_name)

    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./highlighted/' + short_name + '_mask.jpg', mask)

if not os.path.exists('highlighted'):
    os.makedirs('highlighted')
	
files = os.listdir(os.curdir)

images = filter(lambda x: x.endswith('.jpg'), files)
for image in images:
    prepare_mask(image)
