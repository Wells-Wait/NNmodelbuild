# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:58:02 2025

@author: Wells
"""
import numpy as np

class AI:
    def __init__(self,structure):
        self.layers=structure
        
    def run_AI(self,data_point):#one point
        for i in range(0,len(self.layers)):
            data_point=self.layers[i].run_layer(data_point)
        return data_point
    def train_AI(self,x,y):
        #train prep
        for i in range(0,len(self.layers)):
            self.layers[i].train_init()
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
            
            layer_loss=2*(layer_output[-1]-y[point])
            Avrige_Layer_Loss+=(y[point]-layer_output[-1])**2 /number_points
            #back propogate
            for i in range(0,len(self.layers)):
                layer=(len(self.layers)-1)-i
                layer_loss=self.layers[layer].train_layer(layer_loss,layer_output[layer],number_points)
        #commit changes
        for i in range(0,len(self.layers)):
            self.layers[i].train_commit()
        return Avrige_Layer_Loss
class BasicNN:
    def __init__(self,inputs_count,nerons_count,activation):
        self.inputs_count=inputs_count
        self.nerons_count=nerons_count
        self.af=activation
        self.weights=np.ones((nerons_count,inputs_count))#input shape = inputes,1
        self.biases=np.ones(nerons_count)
    def AF(self,array,af):
        match af:
            case "sigmoid": 
                return 1/(1 + np.e ** (-array))
            case _:
                print("bad activation function")
    def AFP(self,array,af):
        match af:
            case "sigmoid": 
                a=np.e**(-array)
                return a/(1 + a)**2
            case _:
                print("bad activation function")
    def run_layer(self,data_point):
        return self.AF(np.matmul(self.weights,data_point)+self.biases,self.af)
    def train_init(self):
        self.weights_T=np.zeros((self.nerons_count,self.inputs_count))#input shape = inputes,1
        self.biases_T=np.zeros(self.nerons_count)
    def train_layer(self,layer_loss,Previus_layer_out,number_points):
        new_layer_loss=np.zeros(self.inputs_count)
        for N in range(0,self.nerons_count):
            for I in range(0,self.inputs_count):
                w=self.weights[N,I]
                b=self.biases[N]
                x=Previus_layer_out[I]
                
                z=x*w+b
                a=self.AF(z, self.af)
                
                dc_da=layer_loss[N]
                da_dz=self.AFP(z, self.af)
                
                dz_dw=x
                dz_db=1
                dz_dx=w
                self.weights_T[N,I]+=(dc_da*da_dz*dz_dw)/number_points
                self.biases_T[N]+=(dc_da*da_dz*dz_db)/(number_points*self.inputs_count)
                new_layer_loss[I]+=(dc_da*da_dz*dz_dx)/self.nerons_count
        return new_layer_loss
                
    def train_commit(self):
        self.weights-=self.weights_T
        self.biases-=self.biases_T
        



