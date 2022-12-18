import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1-x)

def forwardpass(w,b,x):
  z1 = np.dot(x,w[0]) + b[0]
  a1 = sigmoid(z1)  
  z2 = np.dot(a1, w[1]) + b[1] 
  a2 = sigmoid(z2)  
  output = np.dot(a2, w[2]) + b[2]  

  return output,a1,a2

def backwardpass(weights,x,output,y_star):
  dy = output - y_star
  dw3 = np.dot(dy,a2)
  dbias_3 = np.sum(dy, axis=0)
  dz2 = np.dot(dy, weights[2].T) 
  #print(dw3,db3,dz2)

  d2 = sigmoid_derivative(a2) * dz2
  dw2 = np.dot(a1.T,d2)
  dbias_2 = np.sum(d2, axis = 0)
  dz1 = np.dot(d2, weights[1].T)
  #print(dz1)

  d1 = sigmoid_derivative(a1) * dz1 
  dw1 = np.dot(X.T,d1)
  dbias_1 = np.sum(d1, axis=0)
  
  return dw3,dbias_3,dw2,dbias_2,dw1,dbias_1

weights = np.array([[-2,2],[-3,3]]), np.array([[-2,2],[-3,3]]), np.array([[2],[-1.5]]) 
biases = np.array([-1, 1]), np.array([-1, 1]), np.array([-1])
X = np.array([[1,1]]) 
output,a1,a2 = forwardpass(weights,biases,X)
print(output,a1,a2)
y_star = 1 
dw3,db3,dw2,db2,dw1,db1 = backwardpass(weights,X,output,a1,a2,y_star)
print(dw3,db3,dw2,db2,dw1,db1)

bank_train = np.loadtxt("/content/gdrive/My Drive/bank-note/train.csv", delimiter=",")
bank_test = np.loadtxt("/content/gdrive/My Drive/bank-note/test.csv", delimiter=",")

x_train = bank_train[:,0:4]
y_train = bank_train[:,-1]
x_test = bank_test[:,0:4]
y_test = bank_test[:,-1]

for i in range(len(y_train)):
  if y_train[i] == 0:
    y_train[i] = -1
for i in range(len(y_test)):
  if y_test[i] == 0:
    y_test[i] = -1
    
width = 5
weights_rand = [
  np.random.randn(x_train.shape[1], width),
  np.random.randn(width, width),
  np.random.randn(width, 1)
]
biases_rand = [
    np.random.randn(width),
    np.random.randn(width),
    np.random.randn(1)
]

def SGD(weights_rand,biases_rand,x_train):
  gamma_0 = 0.01
  d = 0.01
  t = 0
  for T in range(100):
    shuffle = np.random.permutation(x_train.shape[0])
    for i in shuffle:
        gamma = gamma_0/(1+(gamma_0/d)*t)
        t = t + 1
        output,a1,a2 = forwardpass(weights_rand,biases_rand,x_train[i])
        dw3, db3, dw2, db2, dw1, db1 = backwardpass(weights_rand,x_train[i],output,a1,a2,1)
        print(dw1)
        weights_rand[0] = weights_rand[0] - gamma * dw1
        weights_rand[1] = weights_rand[1] - gamma * dw2
        weights_rand[2] = weights_rand[2] - gamma * dw3
        biases_rand[0] = biases_rand[0] - gamma * db1
        biases_rand[1] = biases_rand[1] - gamma * db2
        biases_rand[2] = biases_rand[2] - gamma * db3
  return weights_rand

weight_SGD = SGD(weights_rand,biases_rand,x_train)
