# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:08:07 2018

@author: weimeng
"""

import csv
import pandas as pd
import numpy as np

positions=[]
positiondata=[]
la=[]
apfir=[]
coname=[]
AP_position=[]

with open(r'C:/Users/weimeng/Desktop/fingerprinting/UJIndoorLoc/tranning_data_2floor.csv') as csvfile:
    readcsv=csv.reader(csvfile, delimiter=',')
    for row in readcsv:
         positions.append(row)
         positiondata.append(row)

del(positiondata[0])

for i in positions[0]:
    coname.append(i)

for i in range(-1,-3,-1):
    coname.pop()


for i in positions:
    la.append(i[-8])
del(la[0])

laf=list(set(la))

data=pd.DataFrame(positiondata,columns=positions[0],dtype=float)

group_by_position=data.groupby(['LONGITUDE','LATITUDE'])

positionmean=group_by_position.mean()
positionmean.to_csv('C:/Users/weimeng/Desktop/out.csv')



for i in range(520):
    app=positionmean.iloc[:,i]
    app[app>0]=-200
    applist=app.tolist()
    appmax=max(applist)
    if appmax!=-200:
        appi=app[app==appmax]
        reappi=appi.reset_index()
        reappi['APNO']=i+1
        temp_data_d=reappi.values.tolist()
        temp_data=temp_data_d[0]
        AP_position.append(temp_data)
        

with open("C:/Users/weimeng/Desktop/Data/2_floor_ap.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow(["LONGITUDE","LATITUDE","RSSI","APnumber"])
    writer.writerows(AP_position)

        
        

    