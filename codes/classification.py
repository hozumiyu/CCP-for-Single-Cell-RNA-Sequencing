# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:13:24 2023

@author: yutah
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def adjust_train_test(y_train, y_test, train_index, test_index):
    '''
        Adjust training and testing data to ensure there are at least 5 of each in the train and 3 of each in test data
        5 * the average number of samples of each class is sampled
    '''
    np.random.seed(1)
    unique_labels_temp = np.intersect1d(y_train, y_test)
    unique_labels_temp.sort()
    unique_labels  = []
    counter = []
    new_test_index = []
    for l in unique_labels_temp:
        l_train = np.where(l == y_train)[0]
        l_test = np.where(l == y_test)[0]
        if l_train.shape[0] > 5 and l_test.shape[0] > 3:
            unique_labels.append(l)
            new_test_index.append(l_test)   #get the index of the y_test that satisfies the condition
            counter.append(l_train.shape[0])
    new_test_index = np.concatenate((new_test_index))
    new_test_index.sort()
    new_y_test = y_test[new_test_index]   #new y_test
    new_test_index = test_index[new_test_index]
    
    
    new_train_index = []
    avgCount = int(np.ceil(np.mean(counter)))   #sample 5x avgCount
    for l in unique_labels:
        l_train = np.where(l == y_train)[0]
        index = np.random.choice(l_train, 5*avgCount)
        new_train_index.append(index)
    new_train_index = np.concatenate(new_train_index)
    new_train_index.sort()
    new_y_train = y_train[new_train_index]
    new_train_index = train_index[new_train_index]
    return new_y_train, new_y_test, new_train_index, new_test_index

def computeSVC(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mySVC = SVC(random_state =1)
    mySVC.fit(X_train, y_train)
    y_pred = mySVC.predict(X_test)
    return y_pred


def balanced_accuarcy(y_true, y_pred):
    ba = balanced_accuracy_score(y_true, y_pred)
    return ba


def compute5foldClassification(X_ccp, X_pca, y_true, max_state = 10):
    BA_ccp = np.zeros(max_state); BA_pca = np.zeros(max_state)
    for state in range( max_state):
        kf = KFold(n_splits=5, shuffle=True, random_state=state)
        ba_ccp = np.zeros(5); ba_pca = np.zeros(5)
        for i, (train_index, test_index) in enumerate(kf.split(X_ccp)):
            y_train = y_true[train_index]; y_test = y_true[test_index]
            y_train, y_test, train_index, test_index = adjust_train_test(y_train, y_test, train_index, test_index)
            X_ccp_train = X_ccp[train_index]; X_ccp_test = X_ccp[test_index]
            y_pred = computeSVC(X_ccp_train, X_ccp_test, y_train, y_test)
            ba_ccp[i] = balanced_accuarcy(y_true[test_index], y_pred)
            
            X_pca_train = X_pca[train_index]; X_pca_test = X_pca[test_index]
            y_pred = computeSVC(X_pca_train, X_pca_test, y_train, y_test)
            ba_pca[i] = balanced_accuarcy(y_test, y_pred)
        BA_ccp[state] = np.mean(ba_ccp)
        BA_pca[state] = np.mean(ba_pca)
            
            
    return BA_ccp, BA_pca