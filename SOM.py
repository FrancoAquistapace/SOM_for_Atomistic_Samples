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

# @params x_list, w_m_list expect lists
# @out returns eucllidean distance between x_list and w_m_list
def vect_distances(x_list, w_m_list):
    # Component wise operations
    distances_sq = []
    for n in range(len(x_list)):
        col = x_list[n]
        dist_n = col - w_m_list[n]
        dist_n_sq = dist_n ** 2
        distances_sq.append(dist_n_sq)
    return np.sqrt(sum(distances_sq))


# @param x expects a data vector
# @param W expects a weight matrix
def matrix_distance(x,W):
    X = np.ones(shape=(W.shape[1],W.shape[0]))*x
    D = np.sqrt(X@x.transpose() - 2*(x@W).transpose() + np.diagonal(W.transpose()@W))
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
                

        
    def old_find_nodes_and_BMU(self,x):
        #Calculate distances
        d = []
        for m in range(self.outputLayerSize):
            d.append(Euclidean_dist(x,self.W1[:,m]))
        D = np.array(d, dtype=float)
        m = np.amin(D)
        #Return BMU
        return D, d.index(m)


    #This function finds the Best Matching Unit (BMU)
    # @param x expects the input layer
    def find_nodes_and_BMU(self,x):
        # Calculate distances
        D = matrix_distance(x,self.W1)
        return D, np.argmin(D)
    
    
    #This function corrects the weights in W1
    def update(self,x,t):
        D, d_min = self.find_nodes_and_BMU(x)
        m = self.outputLayerSize
        H = np.eye(m,m)*h(self.sigma, t, D, d_min) 
        X = x.reshape((self.inputLayerSize,1))*np.ones((self.inputLayerSize,m))
        new_W1 = self.W1 + eta(self.eta,t)*np.matmul((X-self.W1), H)
        self.W1 = new_W1
        
    def old_update(self,x,t):
        D, d_min = self.old_find_nodes_and_BMU(x)
        m = self.outputLayerSize
        H = np.eye(m,m)*h(self.sigma, t, D, d_min) 
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
            if self.outputLayerSize <= 2:
                for i in range(n):
                    self.old_update(new_data[i,:],i+1)
            else:
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
            # Let's try optimizing using vectorization
            # We'll start by getting a data column for every output neuron,
            # containing the distances asociated with it
            dist_columns = []
            data_columns = []
            for col in data.columns.to_list(): # Get columns from data
                data_columns.append(data[col])
                
            for i in range(self.outputLayerSize): # Iterate over each neuron
                # Get neuron weights
                w = self.W1[:,i]
                dist_columns.append(vect_distances(data_columns,w))
                
            # Now that we have the columns, we just have to find the BMU
            all_distances = pd.concat(dist_columns, axis=1)
            output = all_distances.idxmin(axis=1)
            result = pd.concat([data, output], axis=1)
            return result 
