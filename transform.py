"""
Author:     Cody Hawkins
Date:       2/19/21
Class:      CS6420
Desc:       If user specifies manual registration allow them to choose four points on an the input
            and template image, get the warp_matrix and align the images. If automatic registration
            is selected then an identity matrix is for image alignment.
"""

import cv2 as cv
import numpy as np
import sys
import os
from enum import Enum


points = []
temp_points = []
img_copy = None
temp_copy = None


class Warp(Enum):
    # Enums for cv2 motion types
    TRANSLATION = cv.MOTION_TRANSLATION
    EUCLIDEAN = cv.MOTION_EUCLIDEAN
    AFFINE = cv.MOTION_AFFINE
    HOMOGRAPHY = cv.MOTION_HOMOGRAPHY


def click_event(event, x, y, flags, params):
    global points, img_copy
    # points for input image
    if event == cv.EVENT_LBUTTONDOWN:
        font = cv.FONT_HERSHEY_COMPLEX
        # write (x,y) label of every click
        cv.putText(img_copy, str(x) + " , " + str(y), (x, y), font, 1, (0, 255, 0), 2)
        points.append([x, y])
        cv.imshow("Pick Your Points", img_copy)


def click_event_two(event, x, y, flags, params):
    global temp_points, temp_copy
    # points for template image
    if event == cv.EVENT_LBUTTONDOWN:
        font = cv.FONT_HERSHEY_COMPLEX
        # write (x,y) label of every click
        cv.putText(temp_copy, str(x) + " , " + str(y), (x, y), font, 1, (255, 255, 255), 2)
        temp_points.append([x, y])
        cv.imshow("Pick Template Points", temp_copy)


def get_points_in_order(rect_points):
    # return points in ordered direction
    # top left, top right, bottom right, bottom left
    rect_points = np.asarray(rect_points, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = rect_points.sum(axis=1)
    # top left
    rect[0] = rect_points[np.argmin(s)]
    # bottom right
    rect[2] = rect_points[np.argmax(s)]

    diff = np.diff(rect_points, axis=1)
    # top right
    rect[1] = rect_points[np.argmin(diff)]
    # bottom left
    rect[3] = rect_points[np.argmax(diff)]

    return rect


def grab_points(img, template, cv_type, out_file, path):
    global points, temp_points, img_copy, temp_copy
    try:
        # get points from input image
        img_copy = img.copy()
        cv.imshow("Pick Your Points", img_copy)
        cv.setMouseCallback("Pick Your Points", click_event)
        cv.waitKey(0)
        cv.destroyAllWindows()

        rect_1 = get_points_in_order(points)
        (pt_a, pt_b, pt_c, pt_d) = rect_1

        # get points from template image
        temp_copy = template.copy()
        cv.imshow("Pick Template Points", temp_copy)
        cv.setMouseCallback("Pick Template Points", click_event_two)
        cv.waitKey(0)
        cv.destroyAllWindows()

        rect_2 = get_points_in_order(temp_points)
        (p_1, p_2, p_3, p_4) = rect_2

        if Warp[cv_type].value == cv.MOTION_HOMOGRAPHY:
            # 4 points for homography to create 3x3 warp_matrix
            points1 = np.float32([pt_a, pt_b, pt_c, pt_d])
            points2 = np.float32([p_1, p_2, p_3, p_4])

            warp_mat = cv.getPerspectiveTransform(points1, points2)
        else:
            # 3 points for affine/translation/euclidean matrix
            p1 = np.float32([pt_a, pt_b, pt_c])
            p2 = np.float32([p_1, p_2, p_3])

            warp_mat = cv.getAffineTransform(p1, p2)

    except cv.error as err:
        print(err)
        sys.exit(1)

    # write points to .ecc file
    if out_file is not None:
        out_file = os.path.join(path, out_file)
        with open(out_file, "w") as f:
            for point in warp_mat:
                for p in point:
                    f.write(str(p) + "\n")


def register_image(color_img, temp_img, cv_type, manual, ecc_file_man, ecc_file_in, img_out, path, ep_num):
    warp_matrix = []
    points_from_file = []
    if manual is True:
        # read in warp_matrix from manual selection
        ecc_file_man = os.path.join(path, ecc_file_man)

        file = open(ecc_file_man, "r")

        for item in file:
            points_from_file.append(float(item.strip()))

        file.close()
        # create CV_F32 (3, 3) or (2, 3)
        for i in range(0, len(points_from_file), 3):
            warp_matrix.append([points_from_file[i], points_from_file[i + 1], points_from_file[i + 2]])

        warp_matrix = np.asarray(warp_matrix, dtype=np.float32)

    elif manual is False and ecc_file_in is not None:
        # if warp_matrix file specified read it in
        ecc_file_in = os.path.join(path, ecc_file_in)

        file = open(ecc_file_in, "r")

        for item in file:
            points_from_file.append(float(item.strip()))

        file.close()
        # create CV_F32 (3, 3) or (2, 3)
        for i in range(0, len(points_from_file), 3):
            warp_matrix.append([points_from_file[i], points_from_file[i + 1], points_from_file[i + 2]])

        warp_matrix = np.asarray(warp_matrix, dtype=np.float32)
        # if warp_matrix dimensions do not match cv2 motion type create identity matrix to fit
        if Warp[cv_type].value is cv.MOTION_HOMOGRAPHY and warp_matrix.ndim < 3:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        elif Warp[cv_type].value is not cv.MOTION_HOMOGRAPHY and warp_matrix.ndim > 3:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

    else:
        # create identity matrix for automatic registration
        if Warp[cv_type].value == cv.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

    size = temp_img.shape

    iterations = 50

    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, iterations, ep_num)

    im_aligned = np.zeros_like(color_img)

    template = cv.merge((temp_img, temp_img, temp_img))

    # since color image is getting aligned, you must align all channels
    for i in range(3):

        (_, warp_matrix) = cv.findTransformECC(template[:, :, i], color_img[:, :, i], warp_matrix, Warp[cv_type].value,
                                               criteria, inputMask=None, gaussFiltSize=5)

        if Warp[cv_type].value is cv.MOTION_HOMOGRAPHY:
            im_aligned[:, :, i] = cv.warpPerspective(color_img[:, :, i], warp_matrix, (size[1], size[0]),
                                                     flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
        else:
            im_aligned[:, :, i] = cv.warpAffine(color_img[:, :, i], warp_matrix, (size[1], size[0]),
                                                flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

    gray_aligned = cv.cvtColor(im_aligned, cv.COLOR_BGR2GRAY)

    err_img = temp_img - gray_aligned

    abs_err = cv.absdiff(temp_img, gray_aligned)

    min_v, max_v, min_l, max_l = cv.minMaxLoc(err_img)

    abs_err = abs_err * (255 / max_v)

    cv.imshow("Warped Image", im_aligned)
    cv.imshow("Error Image", abs_err)
    img_out = os.path.join(path, img_out)
    cv.imwrite(img_out, im_aligned)

