# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 22:41:19 2018

@author: weimeng
"""

import csv
import xlrd
from matplotlib import pyplot as plt 
from sklearn.preprocessing import normalize


pp=[]
AP=[]
temp=[]
pp1=[]

with open(r'C:/Users/weimeng/Desktop/fingerprinting/UJIndoorLoc/tranning_data_1floor.csv') as csvfile:
    readcsv=csv.reader(csvfile, delimiter=',')
    for row in readcsv:
         pp.append(row)

for i in pp:
    i.pop()
    i.pop()


workbook = xlrd.open_workbook('C:/Users/weimeng/Desktop/Data/1_floor_AP_arrange.xlsx')
booksheet = workbook.sheet_by_index(0)         

for i in range(11):
    AP.append(booksheet.row_values(i))

for j in range(5):
    AP[10].pop()


for k in range(600,700):
    for i in AP:
        for j in i:
            toint=int(j)
            vectorAP=pp[k][toint-1]
            vector_AP=float(vectorAP)
            if vector_AP==100:
                vector_AP=-100
            temp.append(vector_AP)
        pp1.append(temp)
        temp=[]
    pp1.append([-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100])
    for i in range(3):
        pp1[10].insert(0,-100)

    for i in range(2):
        pp1[10].append(-100) 

    fig, ax = plt.subplots()
    im = ax.imshow(pp1, cmap=plt.get_cmap('gray'), interpolation='nearest',
               vmin=-100, vmax=0)
    fig.colorbar(im)
    plt.show()
    pp1=[]


        


        
    