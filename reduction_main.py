# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:56:48 2023

@author: yutah
"""
import os
import numpy as np
from codes.auxilary import load_X, load_y, preprocess_data
from codes.reduction import computeCCP, computePCA
from codes.constructOutpath import constructCCPpath, constructPCApath
data = 'GSE75748time'
n_components_vec = [300]
max_state = 1


X = load_X(data) ; y = load_y(data)
X =  preprocess_data(X)


#CCP
outpath_ccp = constructCCPpath(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile = '%s_ccp_n%d_state%d'%(data, n_components, state)
        if not os.path.exists(outpath_ccp + outfile + '.npy'):
            X_ccp = computeCCP(X, n_components = n_components, random_state = state)
            np.save(outpath_ccp + outfile + '.npy', X_ccp)

        

#PCA
outpath_pca = constructPCApath(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile = '%s_pca_n%d_state%d'%(data,n_components, state)
        if not os.path.exists(outpath_pca + outfile + '.npy'):
            X_pca = computePCA(X, n_components = n_components, random_state = state)
            np.save(outpath_pca + outfile + '.npy', X_pca)