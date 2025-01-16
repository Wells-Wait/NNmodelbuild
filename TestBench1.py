# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:41:57 2025

@author: Wells
"""

import Version5 as v
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  


a=v.AI([v.BasicCNN((20,20), (16,16), "ReLU",.1),v.BasicCNN((5,5), (1,1), "ReLU",.0001) ,v.Flaten2D((5,5)),v.BasicNN(25, 1, "sigmoid",.1)], "MSE")
a2=v.AI([v.BasicCNN((20,20), (16,16), "ReLU",.1),v.Flaten2D((5,5)),v.BasicNN(25, 1, "sigmoid",.1)], "MSE")

#b=v.AI([v.Flaten2D((20,20)),v.BasicNN(400, 64, "ReLU"),v.BasicNN(64, 32, "ReLU"),v.BasicNN(32, 1, "sigmoid")], "MSE")


alfa=40

x=np.random.rand(alfa,20,20)-.5
y=np.random.rand(alfa,1)-.5

x[0:round(alfa/2),1,1]=1
x[0:round(alfa/2),0,0]=1
x[0:round(alfa/2),2,2]=1
x[0:round(alfa/2),2,0]=1
x[0:round(alfa/2),0,2]=1
y[0:round(alfa/2)]*=0

x[round(alfa/2):,0,0]=1
x[round(alfa/2):,1,0]=1
x[round(alfa/2):,2,0]=1
x[round(alfa/2):,0,1]=1
x[round(alfa/2):,2,1]=1
x[round(alfa/2):,0,2]=1
x[round(alfa/2):,1,2]=1
x[round(alfa/2):,2,2]=1
y[round(alfa/2):]+=1
for i in range(5,15):
    x[i]=np.flip(x[i],np.random.randint(0,2))
    
#%%
t=[]
ti=0
e1=[]
e2=[]
fig, ax = plt.subplots(1,1,figsize=(15,15))
def animate(d):
    global t, ti, e1
    da=np.random.randint(0,round(alfa/2),size=(6))
    da2=np.random.randint(round(alfa/2),alfa,size=(6))
    x2=[]
    y2=[]
    for i in range(0,len(da)):
        x2.append(x[da[i]])
        y2.append(y[da[i]])
        x2.append(x[da2[i]])
        y2.append(y[da2[i]])
    
    e1.append(a.train_AI(x2, y2))  
    e2.append(a2.train_AI(x2, y2)) 
    print(f"\n\n",e1[ti],f"\n",i)
    
    
    t.append(ti)
    ti+=1
    
    ax.clear()
    ax.plot(t,e1,color="blue")
    ax.plot(t,e2,color="red")
    
    
anim = FuncAnimation(fig, animate, 
                     frames = 2, interval = 0) 
fig.show()






















