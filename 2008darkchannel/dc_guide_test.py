from numpy.testing._private.utils import jiffies
import cv2
import sys

from scipy.linalg import solve
from scipy.sparse.linalg import lsqr
from dc_guide import *
from darkchannel import *

#dark channel + 原图 导向滤波
if __name__ == '__main__':
    #读取图片
    #fn = "./video/DS_0003/video_2000.jpg" 
    fn = "F:/JZYY/pic/test/3.jpg"
    #fn2 = "F:/JZYY/pic/dehazing/13.png"
    src = cv2.imread(fn);
    #src2 = cv2.imread(fn2);

    #dark channel
    J = darkchannel(src, 0.9, 0.6);
    Re = np.uint8(np.clip(J *255, 0, 255))       #dc结果
    """
    #rgb通道导向滤波 
    I = Re
    I2 = src2        #原图
    w = 0.5 #0.8
    winsize = 13
    Re = np.empty(src.shape, src.dtype)
    Re[:,:, 0] = process(I[:,:, 0], I2[:,:, 0], winsize, w)
    Re[:,:, 1] = process(I[:,:, 1], I2[:,:, 1], winsize, w)
    Re[:,:, 2] = process(I[:,:, 2], I2[:,:, 2], winsize, w)
    Re = np.uint8(np.clip(Re, 0, 255))
    """
    cv2.imshow('after dehaze', J);
    cv2.imwrite('./pic/3dc96.png', Re);
    cv2.waitKey();
    exit();
