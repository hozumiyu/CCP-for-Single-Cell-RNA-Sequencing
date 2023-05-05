# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:56:48 2023

@author: yutah
"""
import os
import numpy as np
from codes.auxilary import load_X, load_y, preprocess_data
from codes.constructOutpath import constructCCPpath, constructPCApath, constructCCPpathResultsClassification, constructPCApathResultsClassification
from codes.classification import compute5foldClassification
data = 'GSE75748time'
n_components_vec = [300]
max_state = 1


X = load_X(data) ; y = load_y(data)
X =  preprocess_data(X)


#CCP
outpath_ccp = constructCCPpath(data)
outpath_ccp_results = constructCCPpathResultsClassification(data)
outpath_pca = constructPCApath(data)
outpath_pca_results = constructPCApathResultsClassification(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile_ccp = '%s_ccp_n%d_state%d'%(data, n_components, state); outfile_pca = '%s_pca_n%d_state%d'%(data, n_components, state)
        X_ccp = np.load(outpath_ccp + outfile_ccp + '.npy')
        X_pca = np.load(outpath_pca + outfile_pca + '.npy')
        if not os.path.exists(outpath_ccp_results + outfile_ccp + '_ba.npy') or not os.path.exists(outpath_pca_results + outfile_pca + '_ba.npy'):
            BA_ccp, BA_pca = compute5foldClassification(X_ccp, X_pca, y)
            np.save(outpath_ccp_results + outfile_ccp + '_ba.npy', BA_ccp)
            np.save(outpath_pca_results + outfile_pca + '_ba.npy', BA_pca)

        

#PCA