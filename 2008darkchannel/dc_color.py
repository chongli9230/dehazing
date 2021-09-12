import cv2;
import math;
import numpy as np;

def gammaTranform(image,gamma,c=1):         #gamma变换
    h, w = image.shape[0],image.shape[1]
    new_img = np.zeros((h,w),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            new_img[i,j] = c*pow(image[i, j], gamma)
    cv2.normalize(new_img,new_img,0,255,cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)
    return new_img