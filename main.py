#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:08:41 2023

@author: yutaho
"""

import numpy as np
from codes.auxilary import load_X, load_y, preprocess_data
from codes.CCP.algorithm.ccp import CCP
from codes.clustering import computeKMeans

data = 'GSE67835'
n_components = [50, 100, 150, 200, 250, 300]

X = load_X(data)
X = preprocess_data(X)
y = load_y(data)

myCCP = CCP(n_components = 100, random_state = 1)
myCCP.fit(X)
X_ccp = myCCP.transform(X)

LABELS, ARI, NMI, SIL = computeKMeans(X_ccp, y)
yy = np.load('GSE67835_ccp_exp_p2.0_s6.0_n100_state1_labels.npy')