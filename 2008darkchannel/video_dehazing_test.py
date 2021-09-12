import cv2
import sys

import numpy as np;
from scipy.linalg import solve
from scipy.sparse.linalg import lsqr
from numpy.testing._private.utils import jiffies
from video_dehazing import *
from darkchannel import *

if __name__ == '__main__':
    
    #读取视频 导向滤波
    video_path = "F:/JZYY/pic/去雾视频/"
    name = "DS_0003.mp4"
    params = video_to_images(video_path + name) 
    re_params = []
    #J = darkchannel(params[0]);
    #J = np.uint8(np.clip(J *255, 0, 255)) 
    #re_params.append(J)
    for  i in range(0, len(params)):
        J = darkchannel(params[i]);
        J = np.uint8(np.clip(J *255, 0, 255))
        
        #每一帧用前一帧结果做导向图进行滤波
        w = 1.2
        winsize = 20
        Re = np.empty(params[0].shape, params[0].dtype)
        Re[:,:, 0] = process(J[:,:, 0], re_params[i-1][:,:, 0], winsize, w)
        Re[:,:, 1] = process(J[:,:, 1], re_params[i-1][:,:, 1], winsize, w)
        Re[:,:, 2] = process(J[:,:, 2], re_params[i-1][:,:, 2], winsize, w)
        Re = np.uint8(np.clip(Re, 0, 255))
        re_params.append(Re)

        #gamma变换亮度调整
        imgHSV = cv2.cvtColor(J, cv2.COLOR_BGR2HSV)
        channelsHSV = cv2.split(imgHSV)
        channelsHSV[2] = gammaTranform(channelsHSV[2],gamma=0.6) # 只在V通道，即灰度图上进行处理
        channels = cv2.merge(channelsHSV)
        new_J = cv2.cvtColor(channels, cv2.COLOR_HSV2BGR)    
        re_params.append(new_J)
        
    params_to_video(re_params, video_path + "new3" + name)
