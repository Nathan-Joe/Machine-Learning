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
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import normalize

bank_train = np.loadtxt("/content/gdrive/My Drive/bank-note/train.csv", delimiter=",")
bank_test = np.loadtxt("/content/gdrive/My Drive/bank-note/test.csv", delimiter=",")
x_train = bank_train[:,0:4]
y_train = bank_train[:,-1]
x_test = bank_test[:,0:4]
y_test = bank_test[:,-1]

x_train = normalize(x_train,axis=0)
x_test = normalize(x_test,axis=0)

model = keras.models.Sequential()
k_init_relu = keras.initializers.he_normal()
k_init_tanh = keras.initializers.glorot_normal()

for i in range(3):
  model.add(Dense(5,activation='relu',kernel_initializer=k_init_relu))
model.add(Dense(1, activation='sigmoid', kernel_initializer=k_init_relu))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)  

evaluation = model.evaluate(x_test, y_test)
accuracy = evaluation[1]
error = 1 - accuracy
print("Depth:", 3, "Width:", 5, "Error:", error)
