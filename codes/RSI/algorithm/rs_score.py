# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:30:36 2022

@author: yutah
"""

import numpy as np
from codes.RSI.algorithm.util import distance, find_label_index, extract_label_data



def rs_index(rs_score, y, label ):
    #label =  np.unique(rs_score[:, 2])
    rs_index = np.zeros([label.shape[0], 3])
    for index, l in enumerate(label):
        idx = np.where(y == l)[0]
        if idx.shape[0] == 0 :
            rs_index[index, 0] = 0
            rs_index[index, 1] = 0
        else:
            dis = rs_score[idx, 0]; sim = rs_score[idx, 1]
            print(dis)
            rs_index[index, 0] = np.mean(dis)
            rs_index[index, 1] = np.mean(sim)
    rs_index[:, 2] = np.abs(rs_index[:, 0] - rs_index[:, 1])
    return rs_index

def rs_full(X, y, metric = 'euclidean'):
    '''
    Compute the full rs score. and will not perform the case of train-test split
    X: feature matrix. Data by feature
    y: labels, can be cluster label or true label
    Output:
        rs_score. data by 3 matrix
            rs_score[:, 0] contain the residue score
            rs_score[:, 1]  contain the similarity score
            rs_score[:, 2] contain the labels
    '''
    dist = distance(Xtrain = X, Xtest = None, metric = 'euclidean', test=False)
    unique_labels = np.unique(y)   #get the unique labels
    rs_score = np.zeros([X.shape[0], 2])  #initialize the rs_scire
    
    for label in unique_labels:  #iterate over unique_labels
        label_index, not_label_index = find_label_index(ytrain = y, label = label, ytest = None, test = False)
        if label_index.shape[0] >= 1: #only compute if label_index is nonempty
            
            residue, similarity = extract_label_data(dist, label_index, not_label_index,  test = False)
            #computing residue score: extract matching label by not matching label matrix
            if not_label_index.shape[0] >= 1:
                residue = np.sum(residue, axis = 1)  #sum along the rows for the residue score
                residue = residue / np.max(residue)
                rs_score[label_index, 0] = residue
            else:
                rs_score[label_index, 0] = 1
            
            similarity = 1-similarity
            if np.min(similarity) < 1e-6:
                similarity[similarity < 0] = 0
                rs_score[label_index, 1] = np.mean(similarity, axis = 1)
            else:
                rs_score[label_index, 1] = np.mean(similarity, axis = 1)
        else:
            print('No index that matches the label')
            
            
    
    #rs_score[:, 0] = rs_score[:, 0] / np.max(rs_score[:, 0])
    rs_score[:, 1] = rs_score[:, 1] / np.max(rs_score[:, 1])
    return rs_score


def rs(Xtrain, ytrain, Xtest, ytest, metric = 'euclidean'):
    '''
    Compute the full rs score. and will not perform the case of train-test split
    X: feature matrix. Data by feature
    y: labels, can be cluster label or true label
    Output:
        rs_score. data by 3 matrix
            rs_score[:, 0] contain the residue score
            rs_score[:, 1]  contain the similarity score
            rs_score[:, 2] contain the labels
    '''
    dist = distance(Xtrain = Xtrain, Xtest = Xtest, metric = 'euclidean', test=True)
    unique_labels = np.unique(ytrain)   #get the unique labels
    rs_score = np.zeros([Xtest.shape[0], 2])  #initialize the rs_scire
    
    for label in unique_labels:  #iterate over unique_labels
        label_index, not_label_index = find_label_index(ytrain = ytrain, label = label, ytest = ytest, test = True)
        label_test_index = label_index[1]
        if label_test_index.shape[0] >= 1:
            residue, similarity = extract_label_data(dist, label_index, not_label_index,  test = True)
            #computing residue score: extract matching label by not matching label matrix
            if not_label_index.shape[0] >= 1:
                residue = np.sum(residue, axis = 1)  #sum along the rows for the residue score
                residue = residue / np.max(residue)
                rs_score[label_test_index, 0] = residue
            else:
                rs_score[label_test_index, 0] = 1
            
            similarity = 1-similarity
            if np.min(similarity) < 1e-6:
                similarity[similarity < 0] = 0
                rs_score[label_test_index, 1] = np.mean(similarity, axis = 1)
            else:
                rs_score[label_test_index, 1] = np.mean(similarity, axis = 1)
        else:
            print('No index that matches the label')
    
    #rs_score[:, 0] = rs_score[:, 0] / np.max(rs_score[:, 0])
    rs_score[:, 1] = rs_score[:, 1] / np.max(rs_score[:, 1])
    return rs_score

