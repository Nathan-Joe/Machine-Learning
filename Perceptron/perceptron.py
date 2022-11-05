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


def standard_perceptron(x_train,y_train):
  w = np.zeros(5)
  w[-1] = 1
  r = 0.1
  for T in range(10):
    shuffle = np.random.permutation(x_train.shape[0])
    for i in shuffle:
      h = np.dot(x_train[i],w)
      result = np.dot(y_train[i],h)
      if result <= 0:
        w = w + r*np.dot(y_train[i],x_train[i])
  return w

def voted_perceptron(x_train,y_train):
  w = np.zeros(5)
  w[-1] = 1
  r = 0.1
  w_list = []
  c_list = []
  m = 0
  cm = 0
  for T in range(10):
    shuffle = np.random.permutation(x_train.shape[0])
    for i in shuffle:
      h = np.dot(x_train[i],w)
      result = np.dot(y_train[i],h)
      if result <= 0:
        w_list.append(w)
        c_list.append(cm)
        w = w + r*np.dot(y_train[i],x_train[i])
        m = m + 1
        cm = 1
      else:
        cm = cm + 1
  return w_list,c_list

def averaged_perceptron(x_train,y_train):
  w = np.zeros(5)
  w[-1] = 1
  r = 0.1
  a = np.zeros(5)
  for T in range(10):
    shuffle = np.random.permutation(x_train.shape[0])
    for i in shuffle:
      h = np.dot(x_train[i],w)
      result = np.dot(y_train[i],h)
      if result <= 0:
        w = w + r*np.dot(y_train[i],x_train[i])
      a = a + w
  return a

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
    
weight = standard_perceptron(x_train1,y_train1)
std_pred = np.sign(np.dot(x_test1,weight))
error = np.sum(np.abs(std_pred - y_test1))
avg_std_pred = error / 1000
print(weight)
print(avg_std_pred)

weight_list, count_list = voted_perceptron(x_train1,y_train1)
sum = 0
for i in range(len(count_list)):
  weight_X = np.sign(np.dot(x_test1,weight_list[i]))
  sum += (count_list[i]*weight_X)
v_pred = np.sign(sum)
error = np.sum(np.abs(v_pred - y_test1))
avg_v_pred = error / 1000
for i in range(len(count_list)):
  print(weight_list[i])
print(count_list)
print(avg_v_pred)

a = averaged_perceptron(x_train1,y_train1)
a_pred = np.sign(np.dot(x_test1,a))
error = np.sum(np.abs(a_pred - y_test1))
avg_a_pred = error / 1000
print(a)
print(avg_a_pred)
