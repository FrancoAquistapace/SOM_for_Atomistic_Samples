"""
Copyright 2021 Franco Aquistapace

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import numpy as np
import pandas as pd


#This function is the learning rate
# @param t expects an int or float value
# @param max_eta expects a float value as the maximum learning rate
def eta(max_eta, t):
    return max_eta/t

#Function for the Gaussian sigma
def sigma(max_sigma, t):
    return max_sigma/t

# @params x,w_m expect 1D arrays
# @out returns euclidean distance of x-w_m
def Euclidean_dist(x,w_m):
    d_vec = x - w_m
    D = np.sqrt(np.dot(d_vec,d_vec))
    return D

# @param sigma_0 expects the max_sigma of the network
# @param t expects a given iteration step
# @param D expects the values of the output nodes
# @param d_min expects the index of the BMU
def h(sigma_0,t,D,d_min):
    I = np.ones(np.shape(D))
    d = np.power((D-I*D[d_min]), 2)
    coef = -1/(2*sigma(sigma_0, t)**2)
    result = np.exp(d*coef)
    return result


class SOM(object):
    def __init__(self, sigma, eta, size):
        #parameters
        self.sigma = sigma
        self.eta = eta
        # @param size expects a two tuple with the input
        # and output number of neurons
        self.inputLayerSize = size[0]   # X1, X2, X3, ...
        self.outputLayerSize = size[1]  # Y1, Y2, Y3, ...
        
        # build weights of each layer, set to random values
        # look at the interconnection diagram to make sense of this
        # 3x4 matrix for input to output
        self.W1 = \
                np.random.rand(self.inputLayerSize, self.outputLayerSize)
                
    #This function finds the Best Matching Unit (BMU)
    # @param x expects the input layer
    # @param W expects the weight matrix
    def find_nodes_and_BMU(self,x):
        #Calculate distances
        d = []
        for m in range(self.outputLayerSize):
            d.append(Euclidean_dist(x,self.W1[:,m]))
        D = np.array(d, dtype=float)
        m = np.amin(D)
        #Return BMU
        return D, d.index(m)
    
    
    #This function corrects the weights in W1
    def update(self,x,t):
        D, d_min = self.find_nodes_and_BMU(x)
        m = self.outputLayerSize
        H = np.ones((m,m))*h(self.sigma, t, D, d_min)
        X = x.reshape((self.inputLayerSize,1))*np.ones((self.inputLayerSize,m))
        new_W1 = self.W1 + eta(self.eta,t)*np.matmul((X-self.W1), H)
        self.W1 = new_W1
        
    def save_weights(self):
        # save this in order to reproduce our cool network
        np.savetxt("weights.txt", self.W1, fmt="%s")
    
    #This function predicts an individual output based on an input x and trained weights
    #Returns the index of the BMU
    def predict_output(self,x):
        D, d_min = self.find_nodes_and_BMU(x)
        return d_min
    
    
    
    #Now lets define the training process
    def train(self, data):
        # @param data expects an ndarray or DataFrame
        if str(type(data)) == '<class \'numpy.ndarray\'>':
            n = data.shape[0]
            for i in range(n):
                self.update(data[i,:],i+1)
                
        elif str(type(data)) == '<class \'pandas.core.frame.DataFrame\'>':
            n = data.shape[0]
            new_data = data.to_numpy()
            for i in range(n):
                self.update(new_data[i,:],i+1)
            
        
    def predict(self, data):
        # @param data expects an ndarray or a DataFrame
        if str(type(data)) == '<class \'numpy.ndarray\'>':
            n = data.shape[0]
            out_values = np.zeros((n,1))
            for i in range(n):
                output = self.predict_output(data[i,:])
                out_values[i,0] = output
            result = np.hstack((data,out_values))
            return result
                
        elif str(type(data)) == '<class \'pandas.core.frame.DataFrame\'>':
            n = data.shape[0]
            new_data = data.to_numpy()
            out_values = np.zeros((n,1))
            for i in range(n):
                output = self.predict_output(new_data[i,:])
                out_values[i,0] = output
            np_result = np.hstack((new_data,out_values))
            result = pd.DataFrame(data = np_result)
            return result    
