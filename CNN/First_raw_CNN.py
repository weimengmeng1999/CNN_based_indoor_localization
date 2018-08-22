# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:28:56 2018

@author: weimeng
"""
import argparse
from sklearn.preprocessing import scale
parser = argparse.ArgumentParser()
parser.add_argument(
        "-T",
        "--training_ratio",
        help="ratio of training data to overall data: default is 0.90",
        default=0.9,
        type=float)
args = parser.parse_args()
training_ratio = args.training_ratio

import csv
import xlrd
import keras
import numpy as np
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import math 
 
def get_distance(lat0, lng0, lat1, lng1):
    distance=math.sqrt((lng0-lng1)**2 + (lat0-lat1)**2)
    return distance



batch_size = 80
epochs = 25

img_rows, img_cols = 12, 12

totaltrain=[]
ppv=[]
ppb=[]
AP=[]
temp=[]
pp1=[]
totalva=[]
y_traint=[]
x_train_position=[]
x_test_position=[]


with open(r'C:/Users/Administrator/Desktop/UJIndoorLoc/tranning_data_2floor.csv') as csvfile:
    readcsv=csv.reader(csvfile, delimiter=',')
    for row in readcsv:
         ppb.append(row)
         temw=[row[-3],row[-2],row[-1]]
         x_train_position.append(temw)


for i in ppb:
    y_traint.append(i[-1])
    i.pop()
    i.pop()
    i.pop()

with open(r'C:/Users/Administrator/Desktop/UJIndoorLoc/validation_data_2floor.csv') as vafile:
    readva=csv.reader(vafile, delimiter=',')
    for row in readva:
         ppv.append(row)
         temt=[row[-2],row[-1]]
         x_test_position.append(temt)

for i in ppv:
    i.pop()
    i.pop()
    i.pop()

del(y_traint[0])
del(x_train_position[0])
del(x_test_position[0])



workbook = xlrd.open_workbook('C:/Users/Administrator/Desktop/Data/2_floor_AP_arrange.xlsx')
booksheet = workbook.sheet_by_index(0)         

for i in range(12):
    AP.append(booksheet.row_values(i))

for j in range(5):
    AP[11].pop()


for k in range(1,1397):
    for i in AP:
        for j in i:
            toint=int(j)
            vectorAP=ppb[k][toint-1]
            vector_AP=float(vectorAP)
            if vector_AP==100:
                vector_AP=-100
            temp.append(vector_AP)
        pp1.append(temp)
        temp=[]

    for i in range(3):
        pp1[11].insert(0,-100)

    for i in range(2):
        pp1[11].append(-100) 
    totaltrain.append(pp1)
    pp1=[]

for a in range(1,88):
    for i in AP:
        for j in i:
            toint=int(j)
            vectorAP=ppv[a][toint-1]
            vector_AP=float(vectorAP)
            if vector_AP==100:
                vector_AP=-100
            temp.append(vector_AP)
        pp1.append(temp)
        temp=[]

    for i in range(3):
        pp1[11].insert(0,-100)

    for i in range(2):
        pp1[11].append(-100) 
    totalva.append(pp1)
    pp1=[]

totaltrain=np.array(totaltrain)
train_AP_features = np.asarray(totaltrain.astype(float))
x_test=np.array(totalva)
y_traint=np.array(y_traint)


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(y_traint)
list(le.classes_)
le.transform(y_traint)


y_traint=np.asarray(pd.get_dummies(y_traint))

num_classes = y_traint.shape[1]

train_val_split = np.random.rand(len(train_AP_features)) < training_ratio # mask index array
x_train = train_AP_features[train_val_split]
y_train = y_traint[train_val_split]
x_val = train_AP_features[~train_val_split]
y_val =y_traint[~train_val_split]


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
x_val= x_val.reshape(x_val.shape[0], img_rows, img_cols,1)
x_test=x_test.reshape(x_test.shape[0], img_rows, img_cols,1)

input_shape = (img_rows, img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val=x_val.astype('float32')
y_train=y_train.astype('int')
y_val=y_val.astype('int')

x_train /= -100
x_test/=-100
x_val/=-100

print(input_shape)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='elu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, verbose=0)



print('Test loss:', score[0])
print('Test accuracy:', score[1])


disum=0.0
predict_testm = model.predict_classes(x_test).astype('int')
predict_testt=le.inverse_transform(predict_testm)
predict_test=predict_testt.reshape((len(x_test),1))
predict_position=np.hstack((x_test_position,predict_test))
x_train_position=np.array(x_train_position)

for i in predict_position:
    for j in x_train_position:
        if j[2]==i[2]:
            x1=float(j[0])
            y1=float(j[1])
            x2=float(i[0])
            y2=float(i[1])
            print(x1,y1,x2,y2)
            print(get_distance(y1,x1,y2,x2))
            disum=disum+get_distance(y1,x1,y2,x2)
            break

print(disum/len(x_test))


K.clear_session()
tf.reset_default_graph()
