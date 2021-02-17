import cv2 as cv
import numpy as np


def equalize(image_1, image_2):
    try:
        input_img = cv.imread(image_1)
        template = cv.imread(image_2, 0)
        if input_img is not None and template is not None:
            if input_img.shape[:2] == template.shape[:2]:
                image = np.copy(input_img)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                input_hist = cv.equalizeHist(gray)
                temp_hist = cv.equalizeHist(template)
            elif input_img.shape[:2] != template.shape[:2]:
                dims = (input_img.shape[1], input_img.shape[0])
                template = cv.resize(template, dims, interpolation=cv.INTER_CUBIC)
                image = np.copy(input_img)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                input_hist = cv.equalizeHist(gray)
                temp_hist = cv.equalizeHist(template)
    except cv.error as err:
        print(err)

    return input_hist, temp_hist