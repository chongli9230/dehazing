import numpy as np
from numpy import lib
from my_autoencoder import *
import cv2


def walkFile(file):
    path = []
    for root, dirs, files in os.walk(file):
        # 遍历文件
        for f in files:
            path.append(os.path.join(root, f))
    return path       

def hist(image):        #计算直方图并归一化
    b, g, r = cv2.split(image) 

    hB = cv2.calcHist([b],[0],None,[256],[0,256])
    hG = cv2.calcHist([g],[0],None,[256],[0,256])
    hR = cv2.calcHist([r],[0],None,[256],[0,256])
    
    histB = ((hB - hB.min())/(hB.max()-hB.min()))
    histG = ((hG - hG.min())/(hG.max()-hG.min()))
    histR = ((hR - hR.min())/(hR.max()-hR.min()))

    Imghist = np.array(list(zip(list(histB),list(histB),list(histB),list(histB),
                list(histG),list(histG),list(histG),list(histG),
                list(histR),list(histR),list(histR),list(histR))))        
    
    #Imghist = Imghist.reshape(256, 12)
    #print(Imghist.shape)   #(256,12)
    return Imghist
"""
#RGB
path1 = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB"
imgpath = walkFile(path1)
all_img = []
for i in imgpath:
    temp = cv2.imread(i)        #BGR
    all_img.append(temp)
#print(all_img)

all_hist= []
for i in all_img:
    all_hist.append(hist(i))
print(all_hist)
"""


"""
img_bilater = cv2.bilateralFilter(img,9,75,75)
cv2.imshow("shuangbian", img_bilater)
cv2.imshow("yuantu", img)
cv2.imwrite("./images/result/2.png", img_bilater)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
