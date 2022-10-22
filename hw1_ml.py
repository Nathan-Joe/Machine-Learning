# -*- coding: utf-8 -*-
"""HW1_ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-yYtUlriuU0S9Ft5SeKM1TCgZ6tJbsYV
"""

import random
import math
import time
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
import statistics
from scipy import linalg as LA
from google.colab import drive
drive.mount("/content/gdrive")

class Node:
    def __init__(self,name):
        self.name = name
        self.leaf = dict()

    def isLeaf(self):
        if len(self.leaf) == 0:
            return True
        else:
            return False

Columns = ['buying','maint','doors','persons','lug_boot','safety','label']

Attributes = {'buying':['vhigh','high','med','low'],
                'maint':['vhigh','high','med','low'],
                'doors':['2','3','4','5more'],
                'persons':['2','4','more'],
                'lug_boot':['small','med','big'],
                'safety':['low','med','high']}

Labels = ['unacc', 'good', 'acc', 'vgood']

S = []
with open("/content/gdrive/My Drive/car_train.csv", 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        S.append(terms)

def best_splits(S,Columns,Attributes,Labels,name):
    attribute_dict = dict()
    for attribute_key,attribute_values in Attributes.items():
        attribute_dict[attribute_key] = Calculate_Gain(S,Columns,attribute_key,attribute_values,Labels,name)
    return max(attribute_dict,key=attribute_dict.get)

def most_common(S,index):
    label_set = set()
    label_dict = dict()
    for example in S:
        val = example[index]
        label_set.add(val)
    for label in label_set:
        label_dict[label] = 0
    for label in label_set:
        for example in S:
            if(example[index] == label):
                label_dict[label] += 1
    return max(label_dict,key=label_dict.get)

def subset(S,Columns,A,atr):
    S_v = []
    for example in S:
        if example[Columns.index(A)] == atr:
            S_v.append(example)
    return S_v

def Calculate_Entropy(S,Labels):
  prob_dict = dict()
  for label in Labels:
    prob_dict[label] = 0
  for label in Labels:
    for example in S:
      if example[len(example)-1] == label:
        prob_dict[label] += 1
  for label,count in prob_dict.items():
    if len(S) == 0:
      prob_dict[label] = 0.0
    else:
      prob_dict[label] = count/len(S)
  entropy = 0.0
  for prob in prob_dict.values():
    if prob == 0.0:
        continue
    entropy = entropy - (prob*math.log2(prob))
  return entropy

def Calculate_ME(S, Labels):
  prob_dict = dict()
  for label in Labels:
    prob_dict[label] = 0
  for label in Labels:
    for example in S:
      if example[len(example)-1] == label:
        prob_dict[label] += 1
  for label,count in prob_dict.items():
    if len(S) == 0:
      prob_dict[label] = 0.0
    else:
      prob_dict[label] = count/len(S)
  majority_error = 1 - max(prob_dict.values())
  return majority_error

def Calculate_Gini(S, Labels):
  prob_dict = dict()
  for label in Labels:
    prob_dict[label] = 0
  for label in Labels:
    for example in S:
      if example[len(example)-1] == label:
        prob_dict[label] += 1
  for label,count in prob_dict.items():
    if len(S) == 0:
      prob_dict[label] = 0.0
    else:
      prob_dict[label] = count/len(S)
  gini = 1.0
  for prob in prob_dict.values():
      gini = gini - (prob**2)
  return gini

def Calculate_Gain(S,Columns,attribute,attribute_values,Labels,name):
  if name == "Entropy":
    gain = Calculate_Entropy(S,Labels)
    for a_value in attribute_values:
        S_v = subset(S,Columns,attribute,a_value)
        gain = gain - (len(S_v)/len(S)*Calculate_Entropy(S_v,Labels))
    return gain

  if name == "ME":
    gain = Calculate_ME(S,Labels)
    for a_value in attribute_values:
        S_v = subset(S,Columns,attribute,a_value)
        gain = gain - (len(S_v)/len(S)*Calculate_ME(S_v,Labels))
    return gain

  if name == "Gini":
    gain = Calculate_Gini(S,Labels)
    for a_value in attribute_values:
        S_v = subset(S,Columns,attribute,a_value)
        gain = gain - (len(S_v)/len(S)*Calculate_Gini(S_v,Labels))
    return gain

def ID3(S,Columns, Attributes, Labels,name,max_depth,current_depth):
  if(len(Labels) == 1):
    leaf_node = str(Labels.pop())
    return Node(leaf_node)

  if(len(Attributes) == 0):
    return Node(str(most_common(S,len(Columns)-1)))
  
  if max_depth == current_depth:
    return Node(str(most_common(S,len(Columns)-1)))

  A = best_splits(S,Columns,Attributes,Labels,name)
  root_node = Node(str(A))
  for atr in Attributes[A]:
    S_v = subset(S,Columns,A,atr)
    if len(S_v) == 0:
        root_node.leaf[atr] = Node(str(most_common(S,len(Columns)-1)))
    else:
        Attributes_v = copy.deepcopy(Attributes)
        Attributes_v.pop(A)
        root_node.leaf[atr] = ID3(S_v,Columns,Attributes_v,Labels,name,max_depth,current_depth+1)
  return root_node

def Predict(example,tree,Columns):
    label = example[len(example)-1]
    current = tree
    while not current.isLeaf():
        decision_attr = current.name 
        attr_val = example[Columns.index(decision_attr)]
        current = current.leaf[attr_val] 
    if current.name == label:
        return True
    else:
        return False

print('Question 2:')
for max_depth in range(1,7):
    me = ID3(S,Columns,Attributes,Labels,"ME",max_depth,0)
    gini = ID3(S,Columns,Attributes,Labels,"Gini",max_depth,0)
    entropy = ID3(S,Columns,Attributes,Labels,"Entropy",max_depth,0)

    train_me = 0
    train_gini = 0
    train_entropy = 0
    train_total = 0

    test_me = 0
    test_gini = 0
    test_entropy = 0
    test_total = 0

    with open("/content/gdrive/My Drive/car_train.csv",'r') as test_file:
        for line in test_file:
            example = line.strip().split(',')
            if Predict(example,me,Columns):
                train_me += 1
            if Predict(example,gini,Columns):
                train_gini += 1
            if Predict(example,entropy,Columns):
                train_entropy += 1
            train_total += 1

    with open("/content/gdrive/My Drive/car_test.csv",'r') as test_file:
        for line in test_file:
            example = line.strip().split(',')
            if Predict(example,me,Columns):
                test_me += 1
            if Predict(example,gini,Columns):
                test_gini += 1
            if Predict(example,entropy,Columns):
                test_entropy += 1
            test_total += 1

    train_me_er = 1-(train_me/train_total)
    train_gini_er = 1-(train_gini/train_total)
    train_entropy_er = 1-(train_entropy/train_total)

    train_entropy_er = 1-(test_me/test_total)
    test_gini_er = 1-(test_gini/test_total)
    test_entropy_er = 1-(test_entropy/test_total)

    print("Train ME depth " + str(max_depth) + ": " + str(train_me_er))
    print("Train Gini depth " + str(max_depth) + ": " + str(train_gini_er))
    print("Train Entropy depth " + str(max_depth) + ": " + str(train_entropy_er) + "\n")
    print("Test ME depth " + str(max_depth) + ": " + str(train_entropy_er))
    print("Test Gini depth "  + str(max_depth) + ": " + str(test_gini_er))
    print("Test Entropy depth "  + str(max_depth) + ": " + str(test_entropy_er) + "\n")

S_train = []
S_test = []

Columns = ['age','job','marital','education','default','balance',
            'housing','loan','contact','day','month','duration',
            'campaign','pdays','previous','poutcome','y']

Attributes = {
    'age':['high','low'],
    'job':['admin.','unknown','unemployed','management','housemaid',
        'entrepreneur','student','blue-collar','self-employed','retired',
        'technician','services'],
    'marital':['married','divorced','single'],
    'education':['unknown','secondary','primary','tertiary'],
    'default':['yes','no'],
    'balance':['high','low'],
    'housing':['yes','no'],
    'loan':['yes','no'],
    'contact':['unknown','telephone','cellular'],
    'day':['high','low'],
    'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
    'duration':['high','low'],
    'campaign':['high','low'],
    'pdays':['high','low'],
    'previous':['high','low'],
    'poutcome':['unknown','other','failure','success']
}

Labels = ['yes','no']

with open('/content/gdrive/My Drive/bank_train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        S_train.append(terms)

with open('/content/gdrive/My Drive/bank_test.csv', 'r') as test_file:
    for line in test_file:
        terms = line.strip().split(',')
        S_test.append(terms)

medians = {'age':0.0,'balance':0.0,'day':0.0,'duration':0.0,'campaign':0.0,
            'pdays':0.0,'previous':0.0}

for A in medians.keys():
    S_a = []
    for example in S_train:
        S_a.append(float(example[Columns.index(A)]))
    medians[A] = statistics.median(S_a)

for A,median in medians.items():
    for example in S_train:
        S_a_values = float(example[Columns.index(A)])
        if S_a_values < median:
            example[Columns.index(A)] = 'low'
        else:
            example[Columns.index(A)] = 'high'

    for example in S_test:
        S_a_values = float(example[Columns.index(A)])
        if S_a_values < median:
            example[Columns.index(A)] = 'low'
        else:
            example[Columns.index(A)] = 'high'

print('Question 3a:')
for max_depth in range(1,17):
    me = ID3(S_train,Columns,Attributes,Labels,"ME",max_depth,0)
    gini = ID3(S_train,Columns,Attributes,Labels,"Gini",max_depth,0)
    entropy = ID3(S_train,Columns,Attributes,Labels,"Entropy",max_depth,0)

    train_me = 0
    train_gini = 0
    train_entropy = 0
    train_total = 0

    test_me = 0
    test_gini = 0
    test_entropy = 0
    test_total = 0

    for example in S_train:
        if Predict(example,me,Columns):
            train_me += 1
        if Predict(example,gini,Columns):
            train_gini += 1
        if Predict(example,entropy,Columns):
            train_entropy += 1
        train_total += 1

    for example in S_test:
      if Predict(example,me,Columns):
          test_me += 1
      if Predict(example,gini,Columns):
          test_gini += 1
      if Predict(example,entropy,Columns):
          test_entropy += 1
      test_total += 1

    train_me_er = 1-(train_me/train_total)
    train_gini_er = 1-(train_gini/train_total)
    train_entropy_er = 1-(train_entropy/train_total)

    train_entropy_er = 1-(test_me/test_total)
    test_gini_er = 1-(test_gini/test_total)
    test_entropy_er = 1-(test_entropy/test_total)

    print("Train ME depth " + str(max_depth) + ": " + str(train_me_er))
    print("Train Gini depth " + str(max_depth) + ": " + str(train_gini_er))
    print("Train Entropy depth " + str(max_depth) + ": " + str(train_entropy_er) + "\n")
    print("Test ME depth " + str(max_depth) + ": " + str(train_entropy_er))
    print("Test Gini depth "  + str(max_depth) + ": " + str(test_gini_er))
    print("Test Entropy depth "  + str(max_depth) + ": " + str(test_entropy_er) + "\n")

def most_common_unknown(S,index):
    label_set = set()
    label_dict = dict()
    for example in S:
        val = example[index]
        if val == 'unknown':
            continue
        else:
            label_set.add(val)
    for label in label_set:
        label_dict[label] = 0
    for label in label_set:
        for example in S:
            if(example[index] == label):
                label_dict[label] += 1
    return max(label_dict,key=label_dict.get)

train_dict = dict()
test_dict = dict()
for attribute in Columns:
    index = Columns.index(attribute)
    train_dict[attribute] = most_common_unknown(S_train,index)
    test_dict[attribute] = most_common_unknown(S_test,index)

for example in S_train:
    for index in range(0,len(Columns)):
        if example[index] == 'unknown':
            example[index] = train_dict[Columns[index]]

for example in S_test:
    for index in range(0,len(Columns)):
        if example[index] == 'unknown':
            example[index] = test_dict[Columns[index]]

print('Question 3b:')
for max_depth in range(1,17):
    me = ID3(S_train,Columns,Attributes,Labels,"ME",max_depth,0)
    gini = ID3(S_train,Columns,Attributes,Labels,"Gini",max_depth,0)
    entropy = ID3(S_train,Columns,Attributes,Labels,"Entropy",max_depth,0)

    train_me = 0
    train_gini = 0
    train_entropy = 0
    train_total = 0

    test_me = 0
    test_gini = 0
    test_entropy = 0
    test_total = 0

    for example in S_train:
        if Predict(example,me,Columns):
            train_me += 1
        if Predict(example,gini,Columns):
            train_gini += 1
        if Predict(example,entropy,Columns):
            train_entropy += 1
        train_total += 1

    for example in S_test:
      if Predict(example,me,Columns):
          test_me += 1
      if Predict(example,gini,Columns):
          test_gini += 1
      if Predict(example,entropy,Columns):
          test_entropy += 1
      test_total += 1

    train_me_er = 1-(train_me/train_total)
    train_gini_er = 1-(train_gini/train_total)
    train_entropy_er = 1-(train_entropy/train_total)

    train_entropy_er = 1-(test_me/test_total)
    test_gini_er = 1-(test_gini/test_total)
    test_entropy_er = 1-(test_entropy/test_total)

    print("Train ME depth " + str(max_depth) + ": " + str(train_me_er))
    print("Train Gini depth " + str(max_depth) + ": " + str(train_gini_er))
    print("Train Entropy depth " + str(max_depth) + ": " + str(train_entropy_er) + "\n")
    print("Test ME depth " + str(max_depth) + ": " + str(train_entropy_er))
    print("Test Gini depth "  + str(max_depth) + ": " + str(test_gini_er))
    print("Test Entropy depth "  + str(max_depth) + ": " + str(test_entropy_er) + "\n")
