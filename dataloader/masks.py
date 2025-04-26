import cv2
import numpy as np
import random


def gen_mask(height, width, num_masks='random'):
    mask = np.ones((height, width))
    num_masks = random.randint(3, 5) if num_masks == 'random' else num_masks

    for _ in range(num_masks):
        mask_width = random.randint(75, 175)
        mask_height = random.randint(75, 175)

        mask_x = random.randint(0, width-mask_width)
        mask_y = random.randint(0, height-mask_height)

        cv2.rectangle(mask, (mask_x, mask_y), (mask_x + mask_width, mask_y + mask_height), 0, -1)

    return mask.reshape(256, 256, 1)


if __name__ == '__main__':
    test = cv2.imread('data/data_256_standard/a/airfield/00000001.jpg')
    # cv2.imshow('test', test)
    # cv2.waitKey(0)
    mask = gen_mask(256, 256).reshape(256, 256, 1)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    masked = (test * mask).astype(np.uint8)
    #masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    cv2.imshow('masked', masked)
    cv2.waitKey(0)
