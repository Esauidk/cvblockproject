import numpy as np
import cv2
import os

"""
*** IMPORTANT ***
Change these paths to the absolute paths in your context (pwd)
"""
# Path to your /input_img directory
path = r'/Users/esauabraham/Documents/CSE455/cvblokusproject/TrainingImages/PieceRecognition'
# Path to your /output_img directory
output_dir = r'/Users/esauabraham/Documents/CSE455/cvblokusproject/TrainingImages/ProcessedEverythingColor'

"""
This is a testing program to play around with image processing and colors
"""

# green mask
for i in range(1, 33):
    os.chdir(path)
    img = cv2.imread(str(i) + '.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = cv2.bitwise_not(green_mask)

    os.chdir(output_dir)
    os.mkdir(str(i))
    cv2.imwrite(str(i) + '/' '0.jpg', img)
    cv2.imwrite(str(i) + '/' 'flip0.jpg', cv2.flip(img, 1))

    ver = ['normal', 'flip']
    styles = {'normal': img, 'flip': cv2.flip(img, 1)}
    for style in ver:
        rotate_temp = styles[style]
        for degree in [90, 180, 270]:
            rotate = cv2.rotate(rotate_temp, cv2.ROTATE_90_CLOCKWISE)
            rotate_temp = rotate
            cv2.imwrite(str(i) + '/' + str(degree) + str(style) + '.jpg', rotate)
