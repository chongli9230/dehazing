import cv2;
import math;
import numpy as np;


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


#The closer the Omega defogging ratio parameter is to 1, the higher the defogging degree is
#T0 minimum transmissivity value in order to prevent the j error caused by too small T, we usually set the lower limit value
def TransmissionEstimate(im, A, sz, w):
    omega = w;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz);
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im + mean_b;
    return q;


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray, et, r, eps);
    return t;


#TX is the lower bound of t to ensure that J does not have too much error
def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx);

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def TransmissionEstimate(im, A):
    omega = 0.9;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, 15);
    return transmission

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

if __name__ == '__main__':
    #读取视频 导向滤波
    video_path = "F:/JZYY/pic/去雾视频/"
    name = "DS_0003.mp4"
    params = video_to_images(video_path + name) 
    re_params = []

    Tmax = params[0].shape[0] * params[0].shape[1] * 0.9
    Tmin = params[0].shape[0] * params[0].shape[1] * 0.1

    src = params[0]
    I = src.astype('float64') / 255;
    dark = DarkChannel(I, 15);
    A = AtmLight(I, dark);
    te = TransmissionEstimate(I, A);
    t = TransmissionRefine(src, te);        #优化透射率
    J = Recover(I, t, A, 0.4);                       #恢复后的图像
    J = np.uint8(np.clip(J *255, 0, 255))

    for  i in range(1, len(params)):
        print(i)
        diff = params[i] - params[i-1]
        movenum = (abs(diff) > 0).astype(np.float32)
        num = np.sum(movenum)

        I = params[i].astype('float64') / 255;
        if num < Tmin :
            dark = DarkChannel(I, 15);
            A = AtmLight(I, dark);      #透射率不变
        if num > Tmin and num < Tmax :
            te = TransmissionEstimate(I, A);        #大气光值不变
            t = TransmissionRefine(src, te);        
        if num > Tmax :
            dark = DarkChannel(I, 15);
            A = AtmLight(I, dark);
            te = TransmissionEstimate(I, A);
            t = TransmissionRefine(src, te);        #优化透射率
    
        J = Recover(I, t, A, 0.4);                       #恢复后的图像
        J = np.uint8(np.clip(J *255, 0, 255))

        imgHSV = cv2.cvtColor(J, cv2.COLOR_BGR2HSV)
        channelsHSV = cv2.split(imgHSV)
        channelsHSV[2] = gammaTranform(channelsHSV[2],gamma=0.6) # 只在V通道，即灰度图上进行处理
        channels = cv2.merge(channelsHSV)
        new_J = cv2.cvtColor(channels, cv2.COLOR_HSV2BGR)    
        re_params.append(new_J)

    params_to_video(re_params, video_path + "new4" + name)