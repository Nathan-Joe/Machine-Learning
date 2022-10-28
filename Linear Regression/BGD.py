import random
import math
import time
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from scipy import linalg as LA
#from google.colab import drive
#drive.mount("/content/gdrive")

def BGD(data_x,data_y):
    w = np.zeros(7)
    threshold = 1e-6
    r = 0.01
    count = 0
    BGD_Cost = []
    norm_diff = 1
    while count < 1000 and norm_diff > threshold:
        y = data_y - np.matmul(data_x,w)
        y2 = np.matmul(data_x,w) - data_y
        Jw = 0.5 * np.sum(np.square(y))
        w = w - r * (np.matmul(y2,data_x))
        norm_diff = np.linalg.norm(w)
        BGD_Cost.append(Jw)
        count += 1
        r*= 0.99
    return BGD_Cost, w

def final_test(data_x, data_y, weight):
  y = data_y - np.matmul(data_x,weight)
  J_cost = 0.5 * np.sum(np.square(y))
  return J_cost

C_train = np.loadtxt("/content/gdrive/My Drive/concrete/train.csv", delimiter=",")
C_test = np.loadtxt("/content/gdrive/My Drive/concrete/test.csv", delimiter=",")
train_df = pd.DataFrame(C_train)
test_df = pd.DataFrame(C_test)
train_x = train_df.iloc[0:,0:7]
train_y = train_df.iloc[:,-1]
test_x = test_df.iloc[0:,0:7]
test_y = test_df.iloc[:,-1]

BGD_Cost, weight= BGD(train_x,train_y)
Test_cost = final_test(test_x,test_y,weight)

plt.plot(BGD_Cost)
plt.xlabel('Count')
plt.ylabel('Cost')
plt.show()
print("Learned weight vector:\n" + str(weight))
print("Learning rate r: 0.01, this got times by 0.99 every single loop which decreases the value by little every single loop" )
print("Cost Function value of test data: " + str(Test_cost))
