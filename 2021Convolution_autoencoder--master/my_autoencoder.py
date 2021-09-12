import os
import re
import logging
import numpy as np
from CSBF import *

import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers.core import input_data, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

"""
#显卡调用设置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     # 使用第1块GPU
"""

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
IMAGE_INPUT_SIZE = (256, 12)
BATCH_SIZE = 6


def build_model():
    logging.info('building model')
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    encoder = input_data(shape=(None, 256, 12,1))
	#encoder = input_data(shape=(None, IMAGE_INPUT_SIZE[0], IMAGE_INPUT_SIZE[1],3), data_preprocessing=img_prep)
    encoder = conv_2d(encoder, 16, 3, activation='relu')
    encoder = dropout(encoder, 0.25)  # you can have noisy input instead
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 2, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 2, 3, activation='relu')
    
    #decoder
    decoder = upsample_2d(encoder, 2)
    decoder = conv_2d(decoder, 16, 3, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 1, 3, activation='relu')

    encoded_str = re.search(r', (.*)\)', str(encoder.get_shape)).group(1)
    encoded_size = np.prod([int(o) for o in encoded_str.split(', ')])
    
    #对于RGB图时，乘以3
    original_img_size = np.prod(IMAGE_INPUT_SIZE)
    percentage = round(encoded_size / original_img_size, 2) * 100
    logging.debug('the encoded representation is {}% of the original \
image'.format(percentage))
    
    return regression(decoder, optimizer='adadelta',loss='binary_crossentropy', learning_rate=0.005)

if __name__ == '__main__':
    #RGB
    path1 = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB"
    imgpath = walkFile(path1)
    all_img = []
    for i in imgpath:
        temp = cv2.imread(i)        #BGR
        all_img.append(temp)

    all_hist= []
    for i in all_img:
        all_hist.append(hist(i))
    #print(all_hist)
    
    conv_autencoder = build_model()
    logging.info('training')
    #model = tflearn.DNN(conv_autencoder, tensorboard_verbose=3,
	 #	checkpoint_path=CHECKPOINT_PATH)
    #model = tflearn.models.dnn.DNN(conv_autencoder, tensorboard_verbose=3,
	       #checkpoint_path=CHECKPOINT_PATH)
    model = tflearn.models.dnn.DNN(conv_autencoder, tensorboard_verbose=0)
	
    #monitorCallback = MonitorCallback(api)
    model.fit(all_hist, all_hist, n_epoch=100, shuffle=True, show_metric=True,
              batch_size=BATCH_SIZE, validation_set=0.1, snapshot_epoch=True,
              run_id='conv_autoencoder',callbacks=[])
    model.save('Endoscope_model.tflearn')

