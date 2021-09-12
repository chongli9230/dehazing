from numpy.testing._private.utils import jiffies
import cv2
import sys

from scipy.linalg import solve
from scipy.sparse.linalg import lsqr
from dc_color import *
from darkchannel import *

#dark channel + 颜色恢复
if __name__ == '__main__':
    #读取图片
    #fn = "./video/DS_0003/video_136.jpg" 
    fn = "F:/JZYY/pic/dehazing/9w.png"
    src = cv2.imread(fn);
    
    #dark channel
    J = darkchannel(src);
    J = np.uint8(np.clip(J *255, 0, 255))       #dc结果
    
    #gamma 变换
    imgHSV = cv2.cvtColor(J, cv2.COLOR_BGR2HSV)
    channelsHSV = cv2.split(imgHSV)
    channelsHSV[2] = gammaTranform(channelsHSV[2],gamma=0.6) # 只在V通道，即灰度图上进行处理
    channels = cv2.merge(channelsHSV)
    new_J = cv2.cvtColor(channels, cv2.COLOR_HSV2BGR)
    
    """
    ############该部分只能测小图，不然梯度重建部分会显示内存不足    
    #亮度调整
    #1 有效对比度
    L = src
    [height,width,pixels] = L.shape
    ycrcb_image = cv2.cvtColor(L, cv2.COLOR_BGR2YCR_CB)
    lam = ycrcb_image[:, :, 0].astype('float64').sum()/(height* width* 1.0)
    new_L = ((1 + 2.0* lam/255)*(L.astype('float64') - lam))
    new_L = np.uint8(np.clip(new_L, 0, 255))
    #E = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCR_CB2BGR)

    #2 亮度融合
    #2.1 亮度梯度
    R = new_J
    E = new_L
    F = np.zeros(R.shape, R.dtype)
    Rycrcb = cv2.cvtColor(R, cv2.COLOR_BGR2YCR_CB)
    Eycrcb = cv2.cvtColor(E, cv2.COLOR_BGR2YCR_CB)
    Y1 = Rycrcb[:, :, 0].astype('float64')
    Y2 = Eycrcb[:, :, 0].astype('float64')
    
    c = -1
    counter = 1
    m = rows = Y1.shape[0]
    n = cols = Y1.shape[1]
    A = np.zeros((rows*(cols-1) + cols*(rows-1) + 1, Y1.shape[0]* Y1.shape[1]), dtype='float64')
    c1 = np.zeros((rows*(cols-1) + cols*(rows-1) + 1, 1), dtype='float64')
    c2 = np.zeros((rows*(cols-1) + cols*(rows-1) + 1, 1), dtype='float64')
    
    for i in range(0, (rows-1)):
        for j in range(cols):
            c+=1
            A[c,(j)*m + i] = -1
            A[c,(j)*m + i + 1] = 1
            c1[c, 0] = Y1[i + 1, j] - Y1[i,j]
            c2[c, 0] = Y2[i + 1, j] - Y2[i,j]    
            counter +=1

    for i in range(rows):
        for j in range(0, (cols - 1)):
            c +=1
            A[c,(j)*m + i] = -1
            A[c,(j+1)*m + i ] = 1
            c1[c, 0] = Y1[i ,j+1] - Y1[i,j]
            c2[c, 0] = Y2[i ,j+1] - Y2[i,j]
            counter += 1
    
    c12 = np.maximum(c1, c2)

    c +=1
    A[c, 0] = 1
    c1[c, 0] = Y1[0,0]
    c2[c, 0]= Y1[0,1]
    x1 = lsqr(A, c12)[0].reshape(cols, rows).T
    newY = np.uint8(np.clip(x1, 0, 255))
    F[:, :, 0] = newY
    
    ##########################CrCb
    Rcr = Rycrcb[:, :, 1].astype('float64')
    Ecr = Eycrcb[:, :, 1].astype('float64')
    yta = 128.0
    newcr = (Rcr * abs(Rcr - yta) + Ecr * abs(Ecr - yta)) / (abs(Rcr - yta) + abs(Ecr - yta)+ 0.001)
    newcr = np.uint8(np.clip(newcr, 0, 255))
    mask = (abs(newcr - Rycrcb[:, :, 1]) < 100).astype(np.float32)
    F[:, :, 1] = Rycrcb[:, :, 1]* (1-mask)+ newcr * mask
    #newR = cv2.cvtColor(Rycrcb, cv2.COLOR_YCR_CB2BGR)

    Rcb = Rycrcb[:, :, 2].astype('float64')
    Ecb = Eycrcb[:, :, 2].astype('float64')
    yta = 128.0
    newcb = (Rcb * abs(Rcb - yta) + Ecb * abs(Ecb - yta)) / (abs(Rcb - yta) + abs(Ecb - yta) + 0.001)
    newcb = np.uint8(np.clip(newcb, 0, 255))
    mask = (abs(newcb - Rycrcb[:, :, 2]) < 100).astype(np.float32)
    F[:, :, 2] = Rycrcb[:, :, 2]* (1-mask)+ newcb * mask

    #F[:, :, 0] = Rycrcb[:, :, 0]
    newF = cv2.cvtColor(F, cv2.COLOR_YCR_CB2BGR)
    """
    cv2.imshow('after dehaze', new_J);
    cv2.imwrite('./pic/result/9Wde2_gamma.png', new_J);
    cv2.waitKey();
    exit();
