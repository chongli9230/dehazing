import cv2
import sys
import math

import numpy as np;
from scipy.linalg import solve
from scipy.sparse.linalg import lsqr
from numpy.testing._private.utils import jiffies

def video_to_images(path):      #视频转图片
    cap = cv2.VideoCapture(path) 
    #print(cap.get(5))      #获取帧率
    frame_count = 1
    params = [] 
    success, frame = cap.read() 
    while(success): 
        #print ('Read a new frame: ', success ) 
        params.append(frame) 
        cv2.imwrite("./video/DS_0003/video" + "_%d.jpg" % frame_count, frame) 
        frame_count = frame_count + 1
        success, frame = cap.read() 
    cap.release() 
    return params    

def params_to_video(params, w_path):       #图片转视频     (图片已存在列表中)
    video = cv2.VideoWriter(w_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (960, 540)) #创建视频流对象-格式一
    for i in range(len(params)):
        video.write(params[i])
    video.release()

def images_to_video(r_path, w_path):            #图片转视频     (读取图片并转化为视频)
    pics = []
    for i in range(0,60):
        src = cv2.imread(r_path + str(i) + ".jpg")
        pics.append(src)
    video = cv2.VideoWriter(w_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (960,540)) #创建视频流对象-格式一
    for i in range(len(pics)):
        video.write(pics[i])
    video.release()

def gammaTranform(image,gamma,c=1):         #gamma变换
    h, w = image.shape[0],image.shape[1]
    new_img = np.zeros((h,w),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            new_img[i,j] = c*pow(image[i, j], gamma)
    cv2.normalize(new_img,new_img,0,255,cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)
    return new_img

########################导向滤波
def guideFilter(I, p, winSize, eps, s):
    
    #输入图像的高、宽
    h, w = I.shape[:2]
    
    #缩小图像
    size = (int(round(w*s)), int(round(h*s)))
    
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    
    #缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X*s)), int(round(X*s)))
    
    #I的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    
    #I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I*small_I, small_winSize)
    
    #方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I #方差公式
    
    small_a = var_small_I / (var_small_I + eps)
    small_b = mean_small_I - small_a*mean_small_I
    
    #对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)
    
    #放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)
    
    q = mean_a*I + mean_b
    
    return q
#s采样比例(4),eps 是调整图的模糊程度与边缘检测精度的参数
#s=4, eps=0.12, winSize=13

def process(grayimage,guideimage, r, w):
    ########## Guided filtering
    #0.12 
    eps = 0.12
    winSize = (r, r)       #类似卷积核（数字越大，磨皮效果越好）
    I = grayimage/255.0      #将图像归一化
    p = guideimage/255.0
    s = 1 #步长
    Filter_img = guideFilter(I, p, winSize, eps,s)
    # 保存导向滤波结果
    Filter_img = np.uint8(np.clip(Filter_img *255, 0, 255))  
    #w = 0.5, 0.8
    enhance = cv2.addWeighted(grayimage, 1+w, Filter_img, -w,0) 
    return enhance