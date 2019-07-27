'''
Created on Sep 1, 2018

@author: Ashesh
'''
import numpy as np
import pandas as pd
from csv import reader
from sklearn.model_selection import KFold
import math
'''
X=pd.read_csv("train_int.csv")
X = X.values
y=pd.read_csv("train_theta.csv")
y = y.values
x=pd.read_csv("test_int.csv")
x = x.values
ytheta=pd.read_csv("test_theta.csv")
ytheta = ytheta.values
'''
dataset = pd.read_csv("BlackCurveDataset.csv")
dataset = dataset.values
kf = KFold(n_splits=5, shuffle=True, random_state=None)
i = 0
for train_index, test_index in kf.split(dataset):
    X_train, X_test = dataset[train_index], dataset[test_index]
    print("X_train:", X_train, "X_test:", X_test)
    if(i == 1):         #To set different partition of train and test dataSet
        break
    i +=1
#print(len(X))
#print(len(y))
#print(len(x))
#print(np.amax(y, axis=0))
# scale units
X = X_train[:,[0]]
y = X_train[:,[1]]
x = X_test[:,[0]]
ytheta = X_test[:,[1]]  

#Printing TestSet in file

dataSetFile2 = open("test_set.csv", "r+")
dataSetFile3 = open("test_set_normalised.csv", "r+")
for i in np.nditer(x):
    #print(i)
    dataSetFile2.write(str(i)+"\n")

for i in np.nditer(ytheta):
    #print(i)
    dataSetFile3.write(str(i)+"\n")

X = X/100 # maximum of X array
x=x/100
y = y/20000.00 # max y value in the graph
ytheta=ytheta/20000.00
print(X.shape)
print(X.size)

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
        self.hiddenSize1 = 1000
        self.hiddenSize2 = 1000
        self.hiddenSize3 = 1000
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize1) * np.sqrt(2/self.inputSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize1, self.hiddenSize2) * np.sqrt(2/self.hiddenSize1) # (3x1) weight matrix from hidden to output layer
        self.W3 = np.random.randn(self.hiddenSize2, self.hiddenSize3) * np.sqrt(2/self.hiddenSize2)
        self.W4 = np.random.randn(self.hiddenSize3, self.outputSize) * np.sqrt(1/self.hiddenSize3)

        self.vdW1 = np.zeros((self.inputSize, self.hiddenSize1))
        self.vdW2 = np.zeros((self.hiddenSize1, self.hiddenSize2))
        self.vdW3 = np.zeros((self.hiddenSize2, self.outputSize))
        
        #biases
        self.b1 = np.zeros((1, self.hiddenSize1))
        self.b2 = np.zeros((1, self.hiddenSize2))
        self.b3 = np.zeros((1, self.outputSize))
        
        self.vdb1 = np.zeros((1, self.hiddenSize1))
        self.vdb2 = np.zeros((1, self.hiddenSize2))
        self.vdb3 = np.zeros((1, self.outputSize))
        
    def forward(self, X):
        #forward propagation through our network             
        #self.z = np.dot(X, self.W1) + self.b1  # dot product of X (input) and first set of 3x2 weights
        self.z = np.dot(X, self.W1) 
        self.z2 = self.relu(self.z) # activation function
        #self.z3 = np.dot(self.z2, self.W2) + self.b2 # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(self.z2, self.W2)
        self.a2 = self.relu(self.z3)
        #self.z4 = np.dot(self.a2, self.W3) + self.b3
        self.z4 = np.dot(self.a2, self.W3)
        #o = self.sigmoid(self.z4) # final activation function
        self.a3 = self.relu(self.z4)
        
        self.z5 = np.dot(self.a3, self.W4)
        o = self.z5
        return o 
    
    def relu(self, s):
        return np.maximum(s, 0)

    def relu_derivative(self, x):
        x[x<0] = 0
        x[x>=0] = 1
        return x
    
    def leaky_relu(self, s):
        return np.maximum(s, 0.01*s)

    def leaky_relu_derivative(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx
    
    
    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o, l_rate, momentum):
        # backward propagate through the network
        self.o_error = 2.0 * (y - o) # error in output
        self.o_delta = self.o_error # applying derivative of sigmoid to error
        #self.b3_delta = (1/X.size)*np.sum(self.o_delta, axis=0, keepdims=True)
        
        self.z3_error = self.o_delta.dot(self.W4.T) 
        self.z3_delta = self.z3_error*self.relu_derivative(self.a3)
        
        self.z2_error = self.z3_delta.dot(self.W3.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.relu_derivative(self.a2) # applying derivative of sigmoid to z2 error
        #self.b2_delta = (1/X.size)*np.sum(self.z2_delta, axis=0, keepdims=True)
        
        self.z1_error = self.z2_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error*self.relu_derivative(self.z2)
        #self.b1_delta = (1/X.size)*np.sum(self.z1_delta, axis=0, keepdims=True)
        
        
        #self.vdW1 = momentum*self.vdW1 + (1-momentum)*((X.T.dot(self.z1_delta))*(1/X.size))
        #self.vdb1 = momentum*self.vdb1 + (1-momentum)*self.b1_delta
        
        #self.vdW2 = momentum*self.vdW2 + (1-momentum)*((self.z2.T.dot(self.z2_delta))*(1/X.size))
        #self.vdb2 = momentum*self.vdb2 + (1-momentum)*self.b2_delta
        
        #self.vdW3 = momentum*self.vdW3 + (1-momentum)*((self.a2.T.dot(self.o_delta))*(1/X.size))
        #self.vdb3 = momentum*self.vdb3 + (1-momentum)*self.b3_delta
        
        self.W1 += (l_rate/X.size)*(X.T.dot(self.z1_delta)) # adjusting first set (input --> hidden) weights
        #self.b1 += (l_rate*self.vdb1)

        self.W2 += (l_rate/X.size)*(self.z2.T.dot(self.z2_delta))  # adjusting second set (hidden --> output) weights
        #self.b2 += l_rate*self.b2_delta
        
        self.W3 += (l_rate/X.size)*(self.a2.T.dot(self.z3_delta))
        #self.b3 += l_rate*self.b3_delta

        self.W4 += (l_rate/X.size)*(self.a3.T.dot(self.o_delta))
        
    def train (self, X, y, l_rate, momentum):
        o = self.forward(X)
        self.backward(X, y, o, l_rate, momentum)
    
    def saveWeights(self):
        np.savetxt("w1.txt",self.W1,fmt="%s")
        np.savetxt("w2.txt",self.W2,fmt="%s")
        np.savetxt("w3.txt",self.W3,fmt="%s")
    
    def predict(self):
        print("predicted data:")
        
        print("output:\n"+str(self.forward(x)*20000.00))
        
        #Calculate the loss in prediction/test set
        
        print ("Loss in Test set: " + str(np.mean(np.square(ytheta - self.forward(x)))))
        
        # load and prepare data
        dataSetFile1 = open("predicted.csv", "a")
        
        for i in np.nditer(self.forward(x)):
            print(i)
            dataSetFile1.write(str(i*20000.00)+"\n")
    
    def step_decay(self, epoch):
        initial_lrate = 0.03
        drop = 0.98
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
    #print("Input: \n" + str(X)) 
    #print("Actual Output: \n" + str(y)) 
    #print("Predicted Output: \n" + str(NN.forward(X)))
    print ("Loss in trainSet: " + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
    print("\n")
    lrate_initial = 0.5
    decay_rate = 1
    #l_rate = lrate_initial/(1+(decay_rate*(i+1)))
    #l_rate = NN.step_decay(i+1)
    l_rate = 0.0001
    momentum = 0.9
    print("l_rate: ", l_rate, "epoch_num: ", (i+1))
    NN.train(X, y, l_rate, momentum)
  
NN.saveWeights()
NN.predict()

        
