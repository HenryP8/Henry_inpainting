import cv2
import numpy as np


def gen_mask(height, width):
    mask = np.ones((height, width))

    cv2.rectangle(mask, (50, 50), (200, 200), 0, -1)

    return mask.reshape(256, 256, 1)


if __name__ == '__main__':
    test = cv2.imread('data/data_256_standard/a/airfield/00000001.jpg')
    cv2.imshow('test', test)
    cv2.waitKey(0)
    mask = gen_mask(256, 256).reshape(256, 256, 1)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    masked = (test * mask).astype(np.uint8)
    #masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    cv2.imshow('masked', masked)
    cv2.waitKey(0)
