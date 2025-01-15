# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:38:12 2024

@author: Wells
"""
import numpy as np
import random





class NN:
    def __init__(self,Structure):
        self.Structure=Structure
    
        self.Weights=[]
        self.Bias=[]
        self.NeronVals=[np.zeros(self.Structure[0])]
        
        # Build weights and nerons based on structrue
        
        for i in range(1,len(self.Structure)):
            self.Weights.append(np.ones((self.Structure[i],self.Structure[i-1]))*random.random()/2)
            self.Bias.append(np.ones(self.Structure[i])*random.random()/2)
            self.NeronVals.append(np.zeros(self.Structure[i]))
        
        
    def forwardProp(self,Dragon_Cat):
        Weights=self.Weights
        Bias=self.Bias
        NeronVals=self.NeronVals
        if (Dragon_Cat.size!=self.Structure[0]):
            print("fail 1")
        NeronVals[0]=Dragon_Cat
        for i in range(0, len(self.Structure)-1):
            
            z=np.matmul(Weights[i],NeronVals[i])+Bias[i]
            NeronVals[i+1]=self.ActivationFunc(z)
        return NeronVals[-1]
    def ActivationFunc(self, Z):
        return 1/(1 + np.e ** (-Z))
    def ActivationFuncPrime(self, Z):
        a=np.e**(-Z)
        return a/(1 + a)**2
    def Train(self,dataX,dataY,amount=1):
        self.bsize=len(dataX)
        self.WeightsUpdate=[]
        self.BiasUpdate=[]
        self.NeronValsUpdate=[]
        error=0
        
        self.NeronValsUpdate=[np.zeros((self.bsize,self.Structure[0]))]
        for i in range(1,len(self.Structure)):
            self.WeightsUpdate.append(np.zeros((self.bsize,self.Structure[i],self.Structure[i-1])))
            self.BiasUpdate.append(np.zeros((self.bsize,self.Structure[i])))
            self.NeronValsUpdate.append(np.zeros((self.bsize,self.Structure[i])))
        

            
            
        for dataP in range(0,len(dataX)):
            X=dataX[dataP]
            YTrue=dataY[dataP]
            Y=self.forwardProp(X)
            
            for i in range(0,len(YTrue)):
                self.NeronValsUpdate[-1][dataP,i]=2*(YTrue[i]-Y[i])
                error+=(YTrue[i]-Y[i])**2/(len(dataX)*len(YTrue))
            
            for layer in range(1,len(self.Structure)):
                layer*=-1
                for neronC in range(0,self.Structure[layer]):
                    for neronN in range(0,self.Structure[layer-1]):
                  
                        #summing the exstra biases
                        w,b,n=self.Something(self.NeronVals[layer-1][neronN], self.Weights[layer][neronC,neronN], self.Bias[layer][neronC], self.NeronValsUpdate[layer][dataP,neronC],amount)
                        self.WeightsUpdate[layer][dataP, neronC,neronN]+=w
                        self.BiasUpdate[layer][dataP,neronC]+=b/self.Structure[layer-1]
                        self.NeronValsUpdate[layer-1][dataP,neronN]+=n
        #Making adjustment vector
        
        self.WeightsVect=[]
        self.BiasVect=[]
        self.NeronValsVect=[np.zeros(self.Structure[0])]
        
        
        for i in range(1,len(self.Structure)):
            self.WeightsVect.append(np.ones((self.Structure[i],self.Structure[i-1]))*random.random())
            self.BiasVect.append(np.ones(self.Structure[i])*random.random())
            self.NeronValsVect.append(np.zeros(self.Structure[i]))
        for dataP in range(0,len(dataX)):
            for layer in range(1,len(self.Structure)):
                layer*=-1
                for neronC in range(0,self.Structure[layer]):
                    self.Bias[layer][neronC]+=self.BiasUpdate[layer][dataP,neronC]/len(dataX)
                    for neronN in range(0,self.Structure[layer-1]):
                        self.Weights[layer][neronC,neronN]+=self.WeightsUpdate[layer][dataP, neronC,neronN]/len(dataX)
        return error            
        
    def Something(self,connected_N_Value,Weight,bias,smallval,amount):
 
        
        z=self.ActivationFuncPrime(Weight*connected_N_Value+bias)
        w=connected_N_Value
        NextSmallVal=Weight
        
        WeightV=w*z*smallval * amount
        biasV=1*z*smallval * amount
        NextSmallValV=NextSmallVal*z*smallval * amount
        return (WeightV,biasV,NextSmallValV)
        
        
        
        
        
        
        
        

    
#%%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
x=np.linspace(0, 1.7,5000)
y=np.cos(x**2)/6 +.5


NN=NN([1,2,2,2,1])
NN.forwardProp(np.array([5]))
X=np.zeros((len(x),1))
Y=np.zeros((len(x),1))
for i in range(0, len(x)):
    X[i][0]=x[i]
    Y[i][0]=y[i]
    


   
    
    
fig, ax = plt.subplots(1,2,figsize=(15,15))

modely=[]
a=0
a2=0
ac=1
xm=np.linspace(0, 1.7,20)
def animate(i):
    global a, a2, ac
    
    errorcolor="blue"
    
    ax[0].clear()
    
    if a>.00005:
        for k in range(0,100):
            rx=[]
            ry=[]
            
            for j in range(0,8):
                r=random.randint(0, len(x)-1)
                rx.append(X[r])
                ry.append(Y[r])
            a=NN.Train(rx,ry,2)
            errorcolor="blue"

    elif a>.00001:
        for k in range(0,100):
            rx=[]
            ry=[]
                
            for j in range(0,15):
                r=random.randint(0, len(x)-1)
                rx.append(X[r])
                ry.append(Y[r])
            a=NN.Train(rx,ry,1.5)
            errorcolor="red"
    else:
        for k in range(0,100):
            rx=[]
            ry=[]
                
            for j in range(0,25):
                r=random.randint(0, len(x)-1)
                rx.append(X[r])
                ry.append(Y[r])
            a=NN.Train(rx,ry,1.3)
            errorcolor="green"

    print(a)
    


    
    modely=[]
    for i in xm:
        modely.append(NN.forwardProp(np.array([i]))[0])
    ax[0].plot(x,y,color="green")
    ax[0].plot(xm,modely,color="red")
    
    
    

    ax[1].plot([ac-1,ac],[a2,a],color=errorcolor)
    ac+=1
    a2=a

    
anim = FuncAnimation(fig, animate, 
                     frames = 2, interval = 0) 
fig.show()






#%%
x=np.ones(30)
y=np.ones(30)*.3


NN1=NN([2,8,8,1])
NN1.forwardProp(np.array([5,5]))
X=np.zeros((len(x),2))
Y=np.zeros((len(x),1))
for i in range(0, len(x)):
    X[i]=[x[i]*.1,x[i]*.2]
    Y[i][0]=y[i]
    
for i in range(0,20):
    print(NN1.Train(X,Y))





