# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:41:57 2025

@author: Wells
"""

import Version5 as v
import numpy as np
import matplotlib.pyplot as plt

a=v.AI([v.BasicCNN((20,20), (15,15), "ReLU"), v.Flaten2D((6,6)),v.BasicNN(36, 1, "sigmoid")], "MSE")
b=v.AI([v.Flaten2D((20,20)),v.BasicNN(400, 64, "ReLU"),v.BasicNN(64, 32, "ReLU"),v.BasicNN(32, 1, "sigmoid")], "MSE")




x=np.random.rand(20,20,20)-.5
y=np.random.rand(20,1)-.5

x[0:10,1,1]=1
x[0:10,0,0]=1
x[0:10,2,2]=1
x[0:10,2,0]=1
x[0:10,0,2]=1
y[0:10]*=0

x[10:,0,0]=1
x[10:,1,0]=1
x[10:,2,0]=1
x[10:,0,1]=1
x[10:,2,1]=1
x[10:,0,2]=1
x[10:,1,2]=1
x[10:,2,2]=1
y[10:]+=1
for i in range(5,15):
    x[i]=np.flip(x[i],np.random.randint(0,2))
    
#%%
t=np.linspace(0, 500,50)
e1=[]
e2=[]
for i in range(0,len(t)):
    e1.append(a.train_AI(x, y))  
    e2.append(b.train_AI(x, y))
    print(f"\n\n",e1[i],f"\n",e2[i],f"\n",i)
    
plt.plot(t,e1,color="green",label="CNN")

plt.plot(t,e2,color="red",label="NN")
plt.legend()
