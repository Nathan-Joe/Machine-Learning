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
