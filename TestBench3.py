# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:57:04 2025

@author: Wells
"""

import pandas as pd
import Version6 as v
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  

#datapath="/Users/wellswait/.cache/kagglehub/datasets/oddrationale/mnist-in-csv/versions/2/mnist_train.csv"
datapath="C:/Users/Wells/.cache/kagglehub/datasets/oddrationale/mnist-in-csv/versions/2/mnist_train.csv"

data=pd.read_csv(datapath)
datap=data


a=v.AI([v.BasicCNN((28,28), (23,23), "sigmoid",1),v.Flaten2D((6,6)),v.BasicNN(36, 10, "softmax",1)], "Cross-Entropy")
#a=v.AI([v.BasicCNN((28,28), (28, 28), "sigmoid",.5),v.Flaten2D((1,1))],"MSE")
#a=v.AI([v.Flaten2D((28,28)),v.BasicNN(28**2, 500, "sigmoid",.5),v.BasicNN(500, 100, "sigmoid",.5),v.BasicNN(100, 50, "sigmoid",.5),v.BasicNN(50, 10, "sigmoid",.5)], "MSE")



#%%

for ugi in range(0,100):
    da=np.linspace((ugi*25), (ugi*25)+50-1,50,dtype=int)
    x2=[]
    y2=[]
   
    for i in range(0,len(da)):
        point=np.zeros((28,28))
        gg=1
        for x in range(0,28):
            for y in range(0,28):
                point[x,y]=data.iloc()[da[i],gg]
                gg+=1
       
        x2.append(point/155)
        point=np.zeros(10)
        point[data["label"][da[i]]]=1
        y2.append(point)
    print(a.train_AI(x2, y2).mean())
    
    
#%%

da=np.random.randint(0,len(data),size=(500))
x2=[]
y2=[]
for i in range(0,len(da)):
    point=np.zeros((28,28))
    gg=1
    for x in range(0,28):
        for y in range(0,28):
            point[x,y]=data.iloc()[da[i],gg]
            gg+=1
   
    x2.append(point)
    point=np.zeros(10)
    point[data["label"][da[i]]]+=1
    y2.append(point)
t=0
acc=0
for i in range(0,len(x2)):
    t+=1
    d=a.run_AI(x2[i])
    if d.argmax()==y2[i].argmax():
       acc+=1
    print(acc/t,"\n\n",d.argmax())
print(acc/t)