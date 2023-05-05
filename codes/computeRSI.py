# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:35:06 2023

@author: yutah
"""

from codes.RSI.algorithm.rs_score import rs, rs_full, rs_index
from sklearn.model_selection import KFold
import numpy as np


def computeRSIclustering(X, LABELS, unique_labels):
    max_state = LABELS.shape[0]
    M = X.shape[0]
    RS_SCORE = np.zeros([max_state, M, 2])
    RSI = np.zeros([max_state, unique_labels.shape[0], 3])
    for state in range(max_state):
        label = LABELS[state, :]
        rs_score_temp = rs_full(X, label)
        RS_SCORE[state, :, :] = rs_score_temp
        rsi_temp = rs_index(rs_score_temp, label, unique_labels )
        RSI[state, :, :] = rsi_temp
    return RS_SCORE, RSI


def computeRSItrue(X, y, unique_labels):
    RS_SCORE = rs_full(X, y)
    RSI = rs_index(RS_SCORE, y, unique_labels )
    return RS_SCORE, RSI


def computeRSI5fold(X_ccp, X, y_true, unique_labels, max_state = 10):
    M = X.shape[0]
    RS_SCORE = np.zeros([max_state, M, 3])
    RSI = np.zeros([max_state, unique_labels.shape[0], 3])
    for state in range( max_state):
        kf = KFold(n_splits=5, shuffle=True, random_state=state)
        RS_SCORE_temp = np.zeros([M, 2])
        RSI_temp = np.zeros([unique_labels.shape[0], 3])
        for i, (train_index, test_index) in enumerate(kf.split(X_ccp)):
            y_train = y_true[train_index]; y_test = y_true[test_index]
            X_train = X[train_index]; X_test  = X[test_index]
            rs_score_temp = rs(Xtrain = X_train, ytrain = y_train, Xtest = X_test, ytest = y_test) 
            rsi_temp = rs_index(rs_score_temp, y_test, unique_labels)
            RS_SCORE_temp[test_index, :] = rs_score_temp
            RSI_temp += rsi_temp / 5
            RS_SCORE[state, test_index, 0] = i
        RS_SCORE[state, :, 1:] = RS_SCORE_temp
        RSI[state, :, :] = RSI_temp 
        
    return RS_SCORE, RSI
