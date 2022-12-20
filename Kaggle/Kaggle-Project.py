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
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Perceptron, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
import csv

C_train = pd.read_csv("/content/gdrive/My Drive/income2022f/train_final.csv")
C_test = pd.read_csv("/content/gdrive/My Drive/income2022f/test_final.csv")
C_test_df = C_test.replace('?',np.nan)
C_train_df = C_train.replace('?',np.nan)

df_two = pd.concat([C_train_df, C_train_df])
df_two.isnull().sum().sort_values(ascending=False)
C_train_df = df_two

C_test_df.fillna(method = "ffill", inplace=True) # works better than idxmax
C_train_df.fillna(method = "ffill", inplace=True)

# occ = C_train_df['occupation'].value_counts().idxmax()
# work = C_train_df['workclass'].value_counts().idxmax()
# native = C_train_df['native.country'].value_counts().idxmax()
# C_train_df['occupation'].fillna(occ, inplace=True)
# C_train_df['workclass'].fillna(work, inplace=True)
# C_train_df['native.country'].fillna(native, inplace=True)

# cc_test = C_test_df['occupation'].value_counts().idxmax()
# work_test = C_test_df['workclass'].value_counts().idxmax()
# native_test = C_test_df['native.country'].value_counts().idxmax()
# C_test_df['occupation'].fillna(occ_test, inplace=True)
# C_test_df['workclass'].fillna(work_test, inplace=True)
# C_test_df['native.country'].fillna(native_test, inplace=True)

C_test_df = C_test_df.drop(columns='ID')
C_test_df = C_test_df.drop(columns='race')
C_test_df = C_test_df.drop(columns='sex')
C_test_df = C_test_df.drop(columns='fnlwgt')
#C_test_df = C_test_df.drop(columns='capital.gain')
#C_test_df = C_test_df.drop(columns='capital.loss')
#C_test_df = C_test_df.drop(columns='relationship')
#C_test_df = C_test_df.drop(columns='education')
#C_test_df = C_test_df.drop(columns='native.country')

C_train_df = C_train_df.drop(columns='race')
C_train_df = C_train_df.drop(columns='sex')
C_train_df = C_train_df.drop(columns='fnlwgt')
#C_train_df = C_train_df.drop(columns='capital.gain')
#C_train_df = C_train_df.drop(columns='capital.loss')
#C_train_df = C_train_df.drop(columns='education')
#C_train_df = C_train_df.drop(columns='native.country')

train_x = C_train_df.iloc[:,:-1]
train_y = C_train_df.iloc[:,-1]

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(train_x)
encoder_train_data = encoder.transform(train_x)
encoder_test_data = encoder.transform(C_test_df)

#Adaboost
ada = AdaBoostClassifier(n_estimators=500,random_state=0, base_estimator= tree.DecisionTreeClassifier(max_depth=1))
ada.fit(encoder_train_data,train_y)
results = ada.predict(encoder_test_data)

#Decision tree
dec_tree = tree.DecisionTreeClassifier(criterion='gini')
dec_tree.fit(encoder_train_data,train_y)
result = dec_tree.predict(encoder_test_data)
print(result)

#Random forest
forest = RandomForestClassifier(random_state = 0)
forest.fit(encoder_train_data,train_y)
results_prob = forest.predict_proba(encoder_test_data)
results = results_prob[:,1]
print(results)

#important features
dt = DecisionTreeRegressor()
dt.fit(encoder_train_data,train_y)
results = dt.predict(encoder_test_data)
print(results)
# get importance
importance = dt.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

#Linear Regression
LR = LinearRegression()
LR.fit(encoder_train_data,train_y)
results = LR.predict(encoder_test_data)
print(results)

#Perceptron
Perceptrons = Perceptron(max_iter = 10000, eta0=0.1, random_state=0)
Perceptrons.fit(encoder_train_data,train_y)
results = Perceptrons.predict(encoder_test_data)
print(results)

#Logistic regression
Logic = LogisticRegression(max_iter = 100000)
Logic.fit(encoder_train_data,train_y)
results_prob = Logic.predict_proba(encoder_test_data)
results = results_prob[:,1]
print(results)

#Neural Network
nn = MLPClassifier(hidden_layer_sizes=50,max_iter=100,random_state=1)
nn.fit(encoder_train_data,train_y)
results_prob = nn.predict_proba(encoder_test_data)
results = results_prob[:,1]
print(results)

#Naive Bayes
naive = GaussianNB()
naive.fit(encoder_train_data.toarray(),train_y)
results = naive.predict(encoder_test_data.toarray())
print(results)

#CSV
with open('/content/gdrive/My Drive/income2022f/Logic_Last.csv','w') as results_file:
    writer = csv.writer(results_file)
    writer.writerow(['ID','Prediction'])
    writer.writerows(enumerate(results,1))
