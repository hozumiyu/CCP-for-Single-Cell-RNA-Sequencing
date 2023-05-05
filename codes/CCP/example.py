# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:08:25 2023

@author: yutah
"""
import os 
import numpy as np
from algorithm.ccp import CCP
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

'''
This is a toy example using tcga-pancan data
Download the data at UCI's website https://archive.ics.uci.edu/ml/machine-learning-databases/00401/
unzip the file'
'''

data = pd.read_csv('./toy_data/TCGA-PANCAN-HiSeq-801x20531/data.csv')
X = data.values[:, 1:].astype('float')
y_temp = pd.read_csv('./toy_data/TCGA-PANCAN-HiSeq-801x20531/labels.csv').values[:, 1]

labels = np.unique(y_temp)
map_labels = { labels[i]: i for i in range(labels.shape[0])}
y_true = np.zeros_like(y_temp)
for i in range(y_true.shape[0]):
    y_true[i] = map_labels[y_temp[i]]
y_true = y_true.astype(int)

kf = KFold(n_splits=5)
kf.get_n_splits(X)
YPRED = np.zeros_like(y_true)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train = X[train_index, :]; y_train = y_true[train_index]
    X_test = X[test_index, :]; y_test = y_true[test_index]
    
    myCCP = CCP(n_components = 32, partition_method = 'kmeans')
    X_train_ccp = myCCP.fit_transform(X_train)
    X_test_ccp = myCCP.transform(X_test)
    
    scaler = StandardScaler()
    X_train_ccp =scaler.fit_transform(X_train_ccp)
    X_test_ccp = scaler.transform(X_test_ccp)
    
    myKNN = KNeighborsClassifier()
    myKNN.fit(X_train_ccp, y_train)
    y_pred = myKNN.predict(X_test_ccp)
    YPRED[test_index] = y_pred
    
print('Accuracy of 5 fold is %.4f'%(accuracy_score(y_true, YPRED)))

