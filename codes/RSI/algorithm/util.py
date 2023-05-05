# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:11:47 2023

@author: yutah
"""

import numpy as np
from sklearn.metrics import pairwise_distances

def distance(Xtrain, Xtest = None, metric = 'euclidean', test=False):
    #compute the max-scaled pairwise distance
    if test:
        dist = pairwise_distances(Xtest, Xtrain, metric = metric)
    else:
        dist = pairwise_distances(Xtrain, metric = metric)
    dist = dist / np.max(dist)
    return dist


def find_label_index(ytrain, label, ytest = None, test = False):
    if test:
        label_train_index = np.where(ytrain == label)[0]
        label_test_index = np.where(ytest == label)[0]
        label_index = [label_train_index, label_test_index]
    else:
        label_index = np.where(ytrain == label)[0]
    not_label_index = np.where(ytrain != label)[0]
    
    return label_index, not_label_index

def extract_label_data(dist, label_index, not_label_index,  test = False):
    if test:
        label_train_index, label_test_index = label_index
        dist = dist[label_test_index, :]
        similarity = dist[:, label_train_index]
        residue = dist[:, not_label_index]
    
    else:
        dist = dist[label_index, :]
        similarity = dist[:, label_index]
        residue = dist[:, not_label_index]
    return residue, similarity