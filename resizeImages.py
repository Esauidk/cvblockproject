import os
import cv2

"""
*** IMPORTANT ***
Change these paths to the absolute paths in your context (pwd)
"""
# Path to your /input_img directory
path = r'/Users/esauabraham/Documents/CSE455/cvblokusproject/TrainingImages/Processed'
# Path to your /output_img directory
output_dir = r'/Users/esauabraham/Documents/CSE455/cvblokusproject/TrainingImages/ProcessedResized'


os.chdir(path)
for dir in os.listdir():
    if os.path.isdir(path + "/" + dir):
        os.chdir(path + "/" + dir)
        for file in os.listdir():
            if os.path.isfile(path + "/" + dir + "/" + file):
                img = cv2.imread(file)
                resized = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
                os.chdir(output_dir)
                if(not os.path.isdir(dir)):
                    os.mkdir(dir)
                os.chdir(output_dir + "/" + dir)
                cv2.imwrite(file, resized)
                os.chdir(path + "/" + dir)
