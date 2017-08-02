import cv2
import os


def find_surnames(file_name):
    print('Working with file' + str(file_name))

    img = cv2.imread(file_name)
    img_final = cv2.imread(file_name)

    short_name = os.path.splitext(file_name)[0]

    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 1))
    dilated = cv2.dilate(new_img, kernel, iterations=16)

    image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    index = 0
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if w > 150 and h > 50:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cropped = img_final[y:y + h, x: x + w]
            cv2.imwrite('./cropped/' + short_name + '_crop_' + str(index) + '.png', cropped)
            index += 1

    cv2.imwrite('./highlighted/' + short_name + '_img.jpg', img)

if not os.path.exists('highlighted'):
	os.makedirs('highlighted')

if not os.path.exists('cropped'):
    os.makedirs('cropped')

files = os.listdir(os.curdir)
images = filter(lambda x: x.endswith('.jpg'), files)
for image in images:
    find_surnames(image)
