import cv2 as cv
import numpy as np
import sys
import os
from enum import Enum


points = []
img_copy = None


class Warp(Enum):
    TRANSLATION = cv.MOTION_TRANSLATION
    EUCLIDEAN = cv.MOTION_EUCLIDEAN
    AFFINE = cv.MOTION_AFFINE
    HOMOGRAPHY = cv.MOTION_HOMOGRAPHY


def click_event(event, x, y, flags, params):
    global points, img_copy

    if event == cv.EVENT_LBUTTONDOWN:
        font = cv.FONT_HERSHEY_COMPLEX
        # write (x,y) label of every click
        cv.putText(img_copy, str(x) + " , " + str(y), (x, y), font, 1, (0, 255, 0), 2)
        points.append([x, y])
        cv.imshow("Pick Your Points", img_copy)


def grab_points(img, cv_type, out_file, path):
    global points, img_copy
    try:
        points1 = []
        img_copy = img.copy()
        cv.imshow("Pick Your Points", img_copy)
        cv.setMouseCallback("Pick Your Points", click_event)
        cv.waitKey(0)
        cv.destroyAllWindows()

        pt_a = points[0]
        pt_b = points[1]
        pt_c = points[2]
        pt_d = points[3]

        # TODO: fix manual point picking, need to test if points from second pic will work for points 2

        # L2 Norm
        width_a = np.sqrt(((pt_a[0] - pt_d[0]) ** 2) + ((pt_a[1] - pt_d[1]) ** 2))
        width_b = np.sqrt(((pt_b[0] - pt_c[0]) ** 2) + ((pt_b[1] - pt_c[1]) ** 2))
        maxWidth = max(int(width_a), int(width_b))

        height_a = np.sqrt(((pt_a[0] - pt_b[0]) ** 2) + ((pt_a[1] - pt_b[1]) ** 2))
        height_b = np.sqrt(((pt_c[0] - pt_d[0]) ** 2) + ((pt_c[1] - pt_d[1]) ** 2))
        maxHeight = max(int(height_a), int(height_b))

        if Warp[cv_type].value == cv.MOTION_HOMOGRAPHY:
            # 4 points for homography to create 3x3 warp_matrix
            points1 = np.float32([pt_a, pt_b, pt_c, pt_d])
            points2 = np.float32([[0, 0], [maxHeight - 1, 0], [0, maxWidth - 1], [maxHeight - 1, maxWidth - 1]])

            warp_mat = cv.getPerspectiveTransform(points1, points2)
        else:
            # 3 points for affine/translation/euclidean matrix
            p1 = np.float32([pt_a, pt_b, pt_c])
            p2 = np.float32([[0, 0], [maxHeight - 1, 0], [0, maxWidth - 1]])

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

    return [maxWidth, maxHeight]


def register_image(color_img, temp_img, cv_type, manual, ecc_file_man, ecc_file_in, img_out, path, ep_num, maxes):
    if manual is True:
        warp_matrix = []
        points_from_file = []
        ecc_file_man = os.path.join(path, ecc_file_man)

        file = open(ecc_file_man, "r")

        for item in file:
            points_from_file.append(float(item.strip()))

        file.close()

        for i in range(0, len(points_from_file), 3):
            warp_matrix.append([points_from_file[i], points_from_file[i + 1], points_from_file[i + 2]])

        warp_matrix = np.asarray(warp_matrix, dtype=np.float32)

    elif manual is False and ecc_file_in is not None:
        warp_matrix = []
        points_from_file = []
        ecc_file_in = os.path.join(path, ecc_file_in)

        file = open(ecc_file_man, "r")

        for item in file:
            points_from_file.append(float(item.strip()))

        file.close()

        for i in range(0, len(points_from_file), 3):
            warp_matrix.append([points_from_file[i], points_from_file[i + 1], points_from_file[i + 2]])

        warp_matrix = np.asarray(warp_matrix, dtype=np.float32)

    else:
        if Warp[cv_type].value == cv.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

    if manual is True:
        size = maxes
    else:
        size = temp_img.shape

    iterations = 50

    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, iterations, ep_num)

    im_aligned = np.zeros_like(color_img)

    template = cv.merge((temp_img, temp_img, temp_img))

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

    err_img = cv.absdiff(temp_img, gray_aligned)

    min_v, max_v, min_l, max_l = cv.minMaxLoc(err_img)

    err_img = err_img * (255 / max_v)

    cv.imshow("Warped Image", im_aligned)
    cv.imshow("Error Image", err_img)
    # TODO: Write out image to outpath


