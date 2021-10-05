import cv2
import numpy as np


def mask_mouse(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 緑色のHSVの値域1
    # hsv_min = np.array([25, 25, 30])
    # hsv_max = np.array([50, 50, 60])

    # hsv_min = np.array([140, 10, 30])
    # hsv_max = np.array([170, 45, 50])

    hsv_min = np.array([140, 10, 30])
    hsv_max = np.array([175, 50, 50])

    mask = cv2.inRange(img, hsv_min, hsv_max)

    white = np.full(img.shape, 255, dtype=img.dtype)
    background = cv2.bitwise_and(
        white, white, mask=mask
    )  # detected mouse area becomes white

    inv_mask = cv2.bitwise_not(mask)  # make mask for not-mouse area
    extracted = cv2.bitwise_and(img, img, mask=inv_mask)

    masked = cv2.add(extracted, background)

    return masked


def get_mousepos(img):
    # img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

    kernel = np.ones((2, 2), np.uint8)
    img_maskmouse = mask_mouse(img)
    img_gray = cv2.cvtColor(img_maskmouse, cv2.COLOR_BGR2GRAY)
    _, img_binary2 = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)
    img_closing = cv2.morphologyEx(img_binary2, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
        img_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        contours = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contours)
    else:
        return None

    return [x, y, w, h]
