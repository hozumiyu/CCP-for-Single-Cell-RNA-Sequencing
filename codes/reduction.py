# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:53:25 2023

@author: yutah
"""

import numpy as np
from codes.CCP.algorithm.ccp import CCP
from sklearn.decomposition import PCA


def computePCA(X, n_components = 50, random_state = 1):
    myPCA = PCA(n_components = n_components, random_state = random_state)
    X_pca = myPCA.fit_transform(X)
    return X_pca


def computeCCP(X, n_components = 50, random_state = 1):
    myCCP = CCP(n_components = n_components, random_state = random_state)
    myCCP.fit(X)
    X_ccp = myCCP.transform(X)
    return X_ccp
