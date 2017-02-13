import csv
import numpy as np
import cv2
import keras
import keras.models as models

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization,Input, Lambda
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping

import sklearn.metrics as metrics
import json

target_im_wide = 320 #200
target_im_height = 160 #66
num_imgs = 8380*2 #8036*2 # *4 because flip, left, and right

steer_ofst = 0.5

drop_fraction = 0.5

prefix = '.\\my_test_data\\' #.\\data\\'
steer_angle = np.ndarray(num_imgs, dtype=np.double)
imgs = np.ndarray((num_imgs, target_im_height, target_im_wide, 3), dtype=np.uint8)

def getimage(fn_img):
    img = cv2.imread(fn_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.resize(img, (target_im_wide, target_im_height)) 
    return img

with open('.\my_test_data\driving_log.csv', 'r') as csvfile:
    i=0
    reader = csv.DictReader(csvfile)
    for row in reader:
        # original image and data
        steer_angle[i] = float(row['steering'])
        imgs[i] = getimage(prefix + row['center'])
        i+=1
        
        # flipped image and data
        steer_angle[i] = -1.0 * float(row['steering'])
        imgs[i] = cv2.flip(getimage(prefix + row['center']), 1)
        i+=1
        
        # left image (**assumes left steer is positive)
#        steer_angle[i] = float(row['steering']) + steer_ofst
#        fn_img=prefix + row['left']
#        fn_img = fn_img.replace(" ", "") #poorly formatted Udacity data
#        imgs[i] = getimage(fn_img)
#        i+=1 
        
        # right image (**assumes left steer is positive)
#        steer_angle[i] = float(row['steering']) - steer_ofst
#        fn_img=prefix + row['right']
#        fn_img = fn_img.replace(" ", "") #poorly formatted Udacity data
#        imgs[i] = getimage(fn_img)
#        i+=1

print('steer_angle shape: {}',steer_angle.shape)  # steering angles are already normalized
print('images shape: {}', imgs.shape)
assert (steer_angle.shape[0] == imgs.shape[0]), "X (%d) and Y (%d) have unequal length." % (steer_angle.shape[0], imgs.shape[0])

# define model.  This is based on the nvidia autopilot model described in
# paper: https://arxiv.org/pdf/1604.07316.pdf
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,     # normalization
                 input_shape=(target_im_height, target_im_wide,3)))
model.add(Cropping2D(cropping=((60,20),(0,0)))) # cropping: tuple of tuple of int (length 2) 
                                                # How many units should be trimmed off at the 
                                                # beginning and end of the 2 cropping
                                                # dimensions (width, height).
model.add(Convolution2D(24,5,5,
                        border_mode='valid', 
                        activation='relu', 
                        subsample=(2,2)))
model.add(Dropout(drop_fraction))
model.add(Convolution2D(36,5,5,
                        border_mode='valid', 
                        activation='relu', 
                        subsample=(2,2)))
model.add(Dropout(drop_fraction))
model.add(Convolution2D(48,5,5,
                        border_mode='valid',
                        activation='relu',
                        subsample=(2,2)))
model.add(Dropout(drop_fraction))
model.add(Convolution2D(64,3,3,
                        border_mode='valid', 
                        activation='relu', 
                        subsample=(1,1)))
model.add(Dropout(drop_fraction))
model.add(Convolution2D(64,3,3,
                        border_mode='valid', 
                        activation='relu', 
                        subsample=(1,1)))
model.add(Dropout(drop_fraction))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.summary()

model.compile(loss='mse', 
              optimizer='adam', 
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(imgs, steer_angle, nb_epoch=10, validation_split=0.2, callbacks=[early_stopping])

model.save('model.h5')
