import cv2
import numpy as np
import math
import copy
from median_filtering import *

if __name__ == '__main__':
    DarkRadius = 19     
    medianRadius = 95
    p = 0.9   #控制去雾因子，取值[0,1]
    fn = "F:/JZYY/pic/dehazing/2w.png"

    src = cv2.imread(fn)
    I = src.astype('float64')
    dark = DarkChannel(I, DarkRadius)      #暗通道
    A = AtmLight(I, dark)      #全局大气光值

    img_median = cv2.medianBlur(dark.astype(np.uint8), medianRadius)      #中值滤波
    #img_median = cv2.bilateralFilter(tempdark,9,75,75)       #双边滤波
    diff =  abs(img_median.astype('float64') - dark)
    diff_median = cv2.medianBlur(diff.astype(np.uint8), medianRadius)    #中值滤波
    #diff_median = cv2.bilateralFilter(tempdiff,9,75,75)     #双边滤波
    diff2 = dark - diff_median.astype('float64')
    diff2[diff2 < 0] = 0
    F = np.maximum(np.minimum(p * diff2, dark), 0)
    Re = np.zeros(src.shape)

    for ind in range(0, 3):
        Re[:, :, ind] =  A[0, ind] *(I[:, :, ind].astype(np.float64) - F)/ (A[0, ind] - F)

    Re = np.uint8(np.clip(Re, 0, 255))


    cv2.imshow('after dehaze', Re);
    cv2.imwrite('./pic/result/2wmedian.png', Re);
    cv2.waitKey();
    exit();