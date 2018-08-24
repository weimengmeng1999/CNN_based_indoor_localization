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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import math  #calculate the distance between two reference points
 
def get_distance(lat0, lng0, lat1, lng1):
    distance=math.sqrt((lng0-lng1)**2 + (lat0-lat1)**2)
    return distance



batch_size = 80
epochs = 25

img_rows, img_cols = 12, 12 #depends on the numbers of APs in this floor

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


with open(r'...p/UJIndoorLoc/tranning_data_2floor.csv') as csvfile: #read training data in the building 1 floor 2
    readcsv=csv.reader(csvfile, delimiter=',')
    for row in readcsv:
         ppb.append(row) #read every row into a list
         temw=[row[-3],row[-2],row[-1]] 
         x_train_position.append(temw) #the longtitude, latitude, and space ID for the RPs in training data


for i in ppb:
    y_traint.append(i[-1]) #space ID - label
    i.pop() 
    i.pop()
    i.pop()

with open(r'.../UJIndoorLoc/validation_data_2floor.csv') as vafile: #read the test data of building 1 floor 2
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



workbook = xlrd.open_workbook('.../2_floor_AP_arrange.xlsx') #read the AP matrix
booksheet = workbook.sheet_by_index(0)         
#arrange the APs as a 12*12 matrix
for i in range(12):
    AP.append(booksheet.row_values(i))

for j in range(5): #the AP numbers in last row is less than 12 
    AP[11].pop() #delete the empty value


for k in range(1,1397): #read the RSSI value of RPs according to the arrangement of AP matrix
    for i in AP:
        for j in i: 
            toint=int(j)
            vectorAP=ppb[k][toint-1] #the RSSI value of RPs for the appointed AP
            vector_AP=float(vectorAP)
            if vector_AP==100: #transfer the no-signal value in ujiindoor database (-100db) to the weakest signal value (-100db) to initialize                vector_AP=-100
            temp.append(vector_AP)
        pp1.append(temp)
        temp=[]
#using -100 to fill the empty value of AP matrix
    for i in range(3):
        pp1[11].insert(0,-100)

    for i in range(2):
        pp1[11].append(-100) 
    totaltrain.append(pp1)
    pp1=[]

for a in range(1,88): #read the validation data and reprocess it
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
x_test=np.array(totalva) #turn the validation data as test data
y_traint=np.array(y_traint)


from sklearn import preprocessing
le=preprocessing.LabelEncoder() #label the space ID as integer number
le.fit(y_traint)
list(le.classes_)
le.transform(y_traint) 


y_traint=np.asarray(pd.get_dummies(y_traint)) #one_hot encoder 

num_classes = y_traint.shape[1]

#split the training data to be training and validation data
train_val_split = np.random.rand(len(train_AP_features)) < training_ratio # mask index array
x_train = train_AP_features[train_val_split]  #split the RSSI value
y_train = y_traint[train_val_split] #split label
x_val = train_AP_features[~train_val_split]
y_val =y_traint[~train_val_split]


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1) #reshape the training, validation and test data to satisfy the CNN input
x_val= x_val.reshape(x_val.shape[0], img_rows, img_cols,1)
x_test=x_test.reshape(x_test.shape[0], img_rows, img_cols,1)

input_shape = (img_rows, img_cols,1) 

x_train = x_train.astype('float32') #transfer the type of arrays
x_test = x_test.astype('float32')
x_val=x_val.astype('float32')
y_train=y_train.astype('int')
y_val=y_val.astype('int')

x_train /= -100 #initializr the RSSI value 
x_test/=-100
x_val/=-100

print(input_shape)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#CNN model
moodel = Sequential()
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

#configures the model for training
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']) #judge every epochs by using 'accuracy' of classfication

#trains the model for a given number of epochs (iterations on a dataset)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))

#returns the loss value & metrics values for the model in test mode
score = model.evaluate(x_val, y_val, verbose=0)



print('Test loss:', score[0])
print('Test accuracy:', score[1])

#calculate the distance error of predict values and true values by using the test data
disum=0.0
predict_testm = model.predict_classes(x_test).astype('int') #predict the classes (labeled) of test data
predict_testt=le.inverse_transform(predict_testm) #transfer the labeled classes into space ID
predict_test=predict_testt.reshape((len(x_test),1)) #reshape the predict array
predict_position=np.hstack((x_test_position,predict_test)) #concatenate the latitude, logtitude, and predict space ID for test data
x_train_position=np.array(x_train_position) 

for i in predict_position:
    for j in x_train_position:
        if j[2]==i[2]: #find the latitude and longtitude for appointed space ID
            x1=float(j[0]) #latitude for training data
            y1=float(j[1]) #longtitude ...
            x2=float(i[0]) #latitude for test data
            y2=float(i[1]) #longtitude...
            print(x1,y1,x2,y2)
            print(get_distance(y1,x1,y2,x2))
            disum=disum+get_distance(y1,x1,y2,x2) #do the sum for all the distance error 
            break

print('distance error:',disum/len(x_test)) #calculate the average distance error

#clear the model
K.clear_session()
tf.reset_default_graph()
