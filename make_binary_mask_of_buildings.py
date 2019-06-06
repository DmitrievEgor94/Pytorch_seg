import cv2
import numpy as np
import os

GROUND_TRUTH_FOLDER = '/home/x/Dmitriev-semseg/dataset/Potsdam' \
                      '/5_Labels_for_participants/'

FOLDER_TO_SAVE = '/home/x/Dmitriev-semseg/dataset/Potsdam/labels/'

print(np.setdiff1d(os.listdir(FOLDER_TO_SAVE), os.listdir('/home/x/Dmitriev-semseg/dataset/Potsdam/data')))


for file_name in os.listdir(GROUND_TRUTH_FOLDER):
    full_path = GROUND_TRUTH_FOLDER + file_name

    image = cv2.imread(full_path)

    pixels_with_no_buidings = np.any(image !=  np.array((255,0,0)).reshape(1, 1, 3), axis=2)
    pixels_with_buildings = np.all(image == np.array((255,0,0)).reshape(1, 1, 3), axis=2)

    image[pixels_with_no_buidings,:] = 0
    image[pixels_with_buildings, :] = 255

    cv2.imwrite(FOLDER_TO_SAVE+file_name, image)