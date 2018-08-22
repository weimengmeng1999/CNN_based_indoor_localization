# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:35:37 2018

@author: weimeng
"""

import csv
import numpy as np

positions=[]
tempposi=[]
with open(r'C:/Users/weimeng/Desktop/fingerprinting/UJIndoorLoc/tranning_data_2floor.csv') as csvfile:
    readcsv=csv.reader(csvfile, delimiter=',')
    for row in readcsv:
         positions.append(row)
         tempposi.append(row)

del tempposi[0]

for i in tempposi:
    for j in range(9):
        i.pop()

affect_number=[]
AP_sum=0
for i in tempposi:
    for j in i:
        if( j!='100' ):
           AP_sum+=1
    affect_number.append(AP_sum)
    AP_sum=0
    
print(affect_number)



for LATITUDE,group in group_by_LATITUDE:
    print(LATITUDE)
    print(group)

for i in positioins[0]:
    group_by_LATITUDE[positions[0]].sum()
            