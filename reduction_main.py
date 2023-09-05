# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:56:48 2023

@author: yutah
"""
import os, sys
import numpy as np
from codes.auxilary import load_X, load_y, preprocess_data
from codes.reduction import computeCCP, computePCA, computeNMF
from codes.constructOutpath import constructCCPpath, constructPCApath, constructNMFpath


data = sys.argv[1]
n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
max_state = 20
n_components_vec = [int(sys.argv[2])]

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
#only need to compute the 300 components
outpath_pca = constructPCApath(data)
for state in range(1, max_state + 1):
    outfile = '%s_pca_state%d'%(data, state)
    if not os.path.exists(outpath_pca + outfile + '.npy'):
        X_pca = computePCA(X, n_components = 300, random_state = state)
        np.save(outpath_pca + outfile + '.npy', X_pca)
        
outpath_nmf = constructNMFpath(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile = '%s_nmf_n%d_state%d'%(data, n_components, state)
        if not os.path.exists(outpath_nmf + outfile + '.npy'):
            X_nmf = computeNMF(X, n_components = n_components, random_state = state)
            np.save(outpath_nmf + outfile + '.npy', X_nmf)