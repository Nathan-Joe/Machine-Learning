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

bank_train = np.loadtxt("/content/gdrive/My Drive/bank-note/train.csv", delimiter=",")
bank_test = np.loadtxt("/content/gdrive/My Drive/bank-note/test.csv", delimiter=",")

x_train1 = copy.copy(bank_train)
y_train1 = bank_train[:,-1]
x_test1 = copy.copy(bank_test)
y_test1 = bank_test[:,-1]

for i in range(len(x_train1)):
  if x_train1[i][-1] == 0:
    x_train1[i][-1] = 1
for i in range(len(x_test1)):
  if x_test1[i][-1] == 0:
    x_test1[i][-1] = 1
    
for i in range(len(y_train1)):
  if y_train1[i] == 0:
    y_train1[i] = -1
for i in range(len(y_test1)):
  if y_test1[i] == 0:
    y_test1[i] = -1
    
w = np.zeros(5)
w[-1] = 1
a = 0.01
gamma_0 = 0.01
t = 0
C = 500/873
for T in range(100):
  shuffle = np.random.permutation(x_train1.shape[0])
  for i in shuffle:
    h = np.dot(x_train1[i],w)
    result = np.dot(y_train1[i],h)
    if result <= 1:
      new_weight = copy.copy(w)
      new_weight[-1] = 0
      #print(new_weight)
      gamma = gamma_0/(1+(gamma_0/a)*t)
      t = t + 1
      #print(gamma)
      #print(t)
      w = w - gamma * new_weight + gamma * C * len(x_train1) * y_train1[i] * x_train1[i]
      print(w)
    else:
      w = (1-gamma)*w
      
w = np.zeros(5)
w[-1] = 1
a = 0.01
gamma_0 = 0.01
t = 1
C = 700/873
for T in range(100):
  shuffle = np.random.permutation(x_train1.shape[0])
  for i in shuffle:
    h = np.dot(x_train1[i],w)
    result = np.dot(y_train1[i],h)
    if result <= 1:
      new_weight = copy.copy(w)
      new_weight[-1] = 0
      #print(new_weight)
      gamma = gamma_0/(1+t)
      t = t + 1
      #print(gamma)
      #print(t)
      w = w - gamma * new_weight + gamma * C * len(x_train1) * y_train1[i] * x_train1[i]
      print(w)
    else:
      w = (1-gamma)*w
      
pred = np.sign(np.dot(x_train1,w))
error = np.sum(np.abs(pred - y_train1))
error/873/2
