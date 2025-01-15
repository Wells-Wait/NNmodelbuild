# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:43:57 2025

@author: Wells
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:34:27 2025

@author: Wells
"""


import numpy as np

class AI:
    def __init__(self,structure,error_type):
        self.layers=structure
        self.error_type=error_type
    def run_AI(self,data_point):#one point
        for i in range(0,len(self.layers)):
            data_point=self.layers[i].run_layer(data_point)
        return data_point
    def train_AI(self,x,y):

        number_points=len(x)
        Avrige_Layer_Loss=0
        #for each data point
        for point in range(0,number_points):
            
            #run+save data point
            layer_output=[]
            data_point=x[point]
            for i in range(0,len(self.layers)):
                layer_output.append(data_point)
                data_point=self.layers[i].run_layer(data_point)
            layer_output.append(data_point)
            
            layer_loss=lossP(y[point],layer_output[-1],self.error_type)
            Avrige_Layer_Loss+=(y[point]-layer_output[-1])**2 /number_points
            #back propogate
            for i in range(0,len(self.layers)):
                layer=(len(self.layers)-1)-i
                layer_loss=self.layers[layer].train_layer(layer_loss,layer_output[layer],number_points,point)

        return Avrige_Layer_Loss
    
class BasicNN:
    def __init__(self,inputs_count,nerons_count,activation):
        self.inputs_count=inputs_count
        self.nerons_count=nerons_count
        self.af=activation
        self.weights=np.random.rand(nerons_count,inputs_count)-.5#input shape = inputes,1
        self.biases=np.random.rand(nerons_count)-.5

    def run_layer(self,data_point):
        return AF(np.matmul(self.weights,data_point)+self.biases,self.af)
    def train_layer(self,layer_loss,Previus_layer_out,number_points,point_number):
        #prep
        if point_number==0:
            self.weights_T=np.zeros((self.nerons_count,self.inputs_count))#input shape = inputes,1
            self.biases_T=np.zeros(self.nerons_count)
            
        new_layer_loss=np.zeros(self.inputs_count)
        for N in range(0,self.nerons_count):
            for I in range(0,self.inputs_count):
                w=self.weights[N,I]
                b=self.biases[N]
                x=Previus_layer_out[I]
                
                z=x*w+b
                
                dc_da=layer_loss[N]
                da_dz=AFP(z, self.af)
                
                dz_dw=x
                dz_db=1
                dz_dx=w
                self.weights_T[N,I]+=(dc_da*da_dz*dz_dw)/number_points
                self.biases_T[N]+=(dc_da*da_dz*dz_db)/(number_points*self.inputs_count)
                new_layer_loss[I]+=(dc_da*da_dz*dz_dx)/self.nerons_count
        #commit
        if point_number==number_points-1:
            self.weights-=self.weights_T
            self.biases-=self.biases_T
        return new_layer_loss
                

class Flaten2D:
    def __init__(self,shape):
        self.x=shape[0]
        self.y=shape[1]
    def run_layer(self,data_point):
        out=np.zeros(self.x*self.y)
        i=0
        for x in range(0,self.x):
            for y in range(0,self.y):
                out[i]=data_point[x,y]
                i+=1
        return out
    def train_layer(self,layer_loss,Previus_layer_out,number_points,point_number):
        out=np.zeros((self.x,self.y))
        i=0
        for x in range(0,self.x):
            for y in range(0,self.y):
                out[x,y]=layer_loss[i]
                i+=1
        return out

class BasicCNN:
    def __init__(self,input_shape,kernel_shape,activation):
        print("not done")
        self.input_x=input_shape[0]
        self.input_y=input_shape[1]
        
        self.kernel_x=kernel_shape[0]
        self.kernel_y=kernel_shape[1]
        
        self.af=activation
        
        self.weights=np.random.rand(self.input_x*self.input_y)#input shape = inputes,1
        self.biases=np.random.rand(1)

    def run_layer(self,data_point):
        out=np.zeros((self.input_x-self.kernel_x+1,self.input_y-self.kernel_y+1))
        for x in range(0,self.input_x-self.kernel_x+1):
            for y in range(0,self.input_y-self.kernel_y+1):
                kernal=data_point[x:x+self.kernel_x,y:y+self.kernel_y]
                
                i=0
                for j in range(0,self.kernel_x):
                    for k in range(0,self.kernel_y):
                        out[x,y]+=kernal[j,k]*self.weights[i]
                        i+=1
                out[x,y]=AF(out[x,y]+self.biases,self.af)
                
                
        return out
        
        return AF(np.matmul(self.weights,data_point)+self.biases,self.af)

    def train_layer(self,layer_loss,Previus_layer_out,number_points,point_number):
        #prep
        if point_number==0:
            self.weights_T=np.zeros(self.input_x*self.input_y)#input shape = inputes,1
            self.biases_T=np.zeros(1)
            
        new_layer_loss=np.zeros((self.input_x,self.input_y))
        for out_x in range(0,self.input_x-self.kernel_x+1):
            for out_y in range(0,self.input_y-self.kernel_y+1):
                i=0
                for xi in range(0,self.input_x):
                    for yi in range(0,self.input_y):
                        w=self.weights[i]
                        b=self.biases[0]
                        x=Previus_layer_out[xi,yi]
                
                        z=x*w+b
                
                        dc_da=layer_loss[out_x,out_y]
                        da_dz=AFP(z, self.af)
                
                        dz_dw=x
                        dz_db=1
                        dz_dx=w
                        self.weights_T[i]+=(dc_da*da_dz*dz_dw)/(number_points*(self.input_x-self.kernel_x+1)*(self.input_y-self.kernel_y+1)  )                       
                        self.biases_T[0]+=(dc_da*da_dz*dz_db)/(number_points*(self.input_x*self.input_y)*(self.input_x-self.kernel_x+1)*(self.input_y-self.kernel_y+1))
                        new_layer_loss[xi,yi]+=(dc_da*da_dz*dz_dx)/((self.input_x-self.kernel_x+1)*(self.input_y-self.kernel_y+1) )
                        i+=1
        #commit
        if point_number==number_points-1:
            self.weights-=self.weights_T
            self.biases-=self.biases_T
        return new_layer_loss
                


def AF(array,af):
    match af:
        case "sigmoid": 
            return 1/(1 + np.e ** (-array))
        case "gaussian":
            return np.e**(-(array)**2 / 2)
        case "ReLU":
            return np.where(array<0,array*0,array)
        case "linear":
            return array
        case _:
            print("bad activation function")
def AFP(array,af):
    match af:
        case "sigmoid": 
            a=np.e**(-array)
            return a/(1 + a)**2
        case "gaussian":
            return -array*np.e**(-(array)**2 / 2)
        case "ReLU":
            return np.where(array<0,array*0,array*0+1)
        case "linear":
            return array*0+1
        case _:
            print("bad activation function")
def lossP(true,pred,error):
    match error:
        case "MSE": 
            return -2*(true-pred)
        case "MAE":
            for i in range(0, len(true)):
                if pred[i]>true[i]:
                    pred[i]=1
                elif pred[i]==true[i]:
                    pred[i]=0
                else:
                    pred[i]=-1
                return pred

        case _:
            print("bad error function")
