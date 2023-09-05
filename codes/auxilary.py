# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 20:46:18 2022

@author: yutah
"""

import numpy as np
import pandas as pd
import csv
import os


def makeFolder(outpath):
    try:
        os.makedirs(outpath)
    except:
        return
    return

def load_X(data):
    inpath = '../data/%s/'%(data)
    X = pd.read_csv(inpath + '%s_full_X.csv'%(data))
    X = X.values[:, 1:].astype(float)
    return X

def preprocess_data(X):
    X = np.log10(1+X).T
    variance = np.var(X, axis = 0)
    idx = np.where(variance > 1e-6)[0]
    X = X[:, idx]
    return X


def load_y(data):
    inpath = '../data/%s/'%(data)
    y = pd.read_csv(inpath + '%s_full_labels.csv'%(data))
    y = np.array(list(y['Label'])).astype(int)
    return y

def load_ypred(data):
    inpath = './sc3_results/%s/'%(data)
    y = pd.read_csv(inpath + '%s_SC3_test1.csv'%(data))
    k = list(y.keys())
    y = np.array(list(y[k[-1]])).astype(int)
    return y