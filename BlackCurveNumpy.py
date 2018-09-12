'''
Created on Sep 1, 2018

@author: Ashesh
'''
import numpy as np
import pandas as pd
from csv import reader

# X  = (hours sleeping, hours studying), y = score on test
#X X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
#y = np.array(([92], [86], [89]), dtype=float)
X=pd.read_csv("train_int.csv")
X = X.values
y=pd.read_csv("train_theta.csv")
y = y.values
x=pd.read_csv("test_int.csv")
x = x.values
ytheta=pd.read_csv("test_theta.csv")
ytheta = ytheta.values
#print(len(X))
#print(len(y))
#print(len(x))
#print(np.amax(y, axis=0))
# scale units
'''
X = X/np.amax(X, axis=0) # maximum of X array
x=x/np.amax(x,axis=0)
y = y/22000 # max y value in the graph
'''
X = X/np.amax(X, axis=0) # maximum of X array
x=x/np.amax(x,axis=0)
y = y/100 # max y value in the graph
ytheta=ytheta/100
print(X.size)
dataSetFile2 = open("test_set.csv", "r+")
dataSetFile3 = open("test_set_normalised.csv", "r+")
for i in np.nditer(x):
    #print(i)
    dataSetFile2.write(str(i)+"\n")

for i in np.nditer(ytheta):
    #print(i)
    dataSetFile3.write(str(i)+"\n")
    
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 1
        self.outputSize = 1
        self.hiddenSize1 = 10
        self.hiddenSize2 = 10
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize1)*0.01 # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize1, self.hiddenSize2)*0.01 # (3x1) weight matrix from hidden to output layer
        self.W3 = np.random.randn(self.hiddenSize2, self.outputSize)*0.01

        #biases
        self.b1 = np.zeros((1, self.hiddenSize1))
        self.b2 = np.zeros((1, self.hiddenSize2))
        self.b3 = np.zeros((1, self.outputSize))
        
    def forward(self, X):
        #forward propagation through our network             
        self.z = np.dot(X, self.W1) + self.b1  # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) + self.b2 # dot product of hidden layer (z2) and second set of 3x1 weights
        self.a2 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a2, self.W3)
        o = self.sigmoid(self.z4) # final activation function
        return o 

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o, l_rate):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
        self.b3_delta = (1/X.size)*np.sum(self.o_delta, axis=0, keepdims=True)
        
        self.z2_error = self.o_delta.dot(self.W3.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.a2) # applying derivative of sigmoid to z2 error
        self.b2_delta = (1/X.size)*np.sum(self.z2_delta, axis=0, keepdims=True)
        
        self.z1_error = self.z2_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error*self.sigmoidPrime(self.z2)
        self.b1_delta = (1/X.size)*np.sum(self.z1_delta, axis=0, keepdims=True)
        
        self.W1 -= l_rate*(X.T.dot(self.z1_delta))*(1/X.size) # adjusting first set (input --> hidden) weights
        self.b1 -= (l_rate*self.b1_delta)

        self.W2 -= l_rate*(self.z2.T.dot(self.z2_delta))*(1/X.size)  # adjusting second set (hidden --> output) weights
        self.b2 -= l_rate*self.b2_delta
        
        self.W3 -= l_rate*(self.a2.T.dot(self.o_delta))*(1/X.size)
        self.b3 -= l_rate*self.b3_delta
        
    def train (self, X, y, l_rate):
        o = self.forward(X)
        self.backward(X, y, o, l_rate)
    
    def saveWeights(self):
        np.savetxt("w1.txt",self.W1,fmt="%s")
        np.savetxt("w2.txt",self.W2,fmt="%s")
        np.savetxt("w3.txt",self.W3,fmt="%s")
    
    def predict(self):
        print("predicted data:")
        #print("input:\n"+str(x))
        print("output:\n"+str(self.forward(x)*100))
        
        print ("Loss in Test set: " + str(np.mean(np.square(ytheta - self.forward(x)))))
        
        dSetFile = open("trained_theta.csv", "a")
        for i in np.nditer(self.forward(X)):
            print(i)
            dSetFile.write(str(i*100)+"\n")
        
        # load and prepare data
        dataSetFile1 = open("predicted.csv", "a")
        
        for i in np.nditer(self.forward(x)):
            print(i)
            dataSetFile1.write(str(i*100)+"\n")


NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
    #print("Input: \n" + str(X)) 
    #print("Actual Output: \n" + str(y)) 
    #print("Predicted Output: \n" + str(NN.forward(X)))
    print ("Loss: " + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
    print("\n")
    l_rate = 0.01
    NN.train(X, y, l_rate)
  
NN.saveWeights()
NN.predict()

        