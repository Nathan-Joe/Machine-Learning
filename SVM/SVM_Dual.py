import random
import math
import time
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from scipy import linalg as LA
from google.colab import drive
drive.mount("/content/gdrive")
from scipy.optimize import minimize

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
    
def dual_objective(a,x,y):
  yMatrix = y * np.ones((len(y),len(y)))
  aMatrix = a * np.ones((len(a),len(a)))

  xxT = x@x.T
  yyT = yMatrix*yMatrix.T
  aaT = aMatrix*aMatrix.T

  inner = yyT * aaT * xxT

  return 0.5*np.sum(inner) - np.sum(a)

def dual_objective_gauss(a,x,y):
  yMatrix = y * np.ones((len(y),len(y)))
  aMatrix = a * np.ones((len(a),len(a)))
  gamma = 0.1
  Gauss = np.zeros((x.shape[0],x.shape[0]))

  for i in range (len(x)):
    for j in range(len(x)):
      X_subs = x[i] - x[j]
      Gauss[i,j] = math.exp(-np.linalg.norm(X_subs, ord=2)/gamma)
  yyT = yMatrix*yMatrix.T
  aaT = aMatrix*aMatrix.T

  inner = yyT * aaT * Gauss

  return 0.5*np.sum(inner) - np.sum(a)

C = 100/873
cons = [{'type':'eq','fun': lambda a: np.sum(np.dot(a, y_train))},
        {'type': 'ineq','fun': lambda a : a},
        {'type': 'ineq','fun': lambda a: C - a }]

solution = minimize(dual_objective_gauss, x0 = np.zeros(shape=(len(x_train),)),args=(x_train,y_train),method = 'SLSQP',constraints = cons)

w_star = np.zeros(4)
for i in range(len(x_train)):
  w_star += (a_star[i] * y_train[i]) * x_train[i]
  print(w_star)
  
b_star = 0
for i in range(len(x_train)):
  b_star += y_train[i] - np.dot(w_star, x_train[i])
b_star = b_star / len(x_train)
print(b_star)

pred = np.sign(np.dot(x_train,w_star)+b_star)
error = np.sum(np.abs(pred - y_train))
error/872/2

