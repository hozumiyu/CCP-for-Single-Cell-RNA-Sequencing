# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:56:48 2023

@author: yutah
"""
import os
import numpy as np
from codes.auxilary import load_X, load_y, preprocess_data
from codes.constructOutpath import constructCCPpath, constructPCApath, constructCCPpathResults, constructPCApathResults
from codes.clustering import computeKMeans
data = 'GSE75748time'
n_components_vec = [300]
max_state = 1


X = load_X(data) ; y = load_y(data)
X =  preprocess_data(X)


#CCP
outpath_ccp = constructCCPpath(data)
outpath_ccp_results = constructCCPpathResults(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile = '%s_ccp_n%d_state%d'%(data, n_components, state)
        X_ccp = np.load(outpath_ccp + outfile + '.npy')
        if not os.path.exists(outpath_ccp_results + outfile + '_labels.npy') or not os.path.exists(outpath_ccp_results + outfile + '_ari.npy')\
            or not os.path.exists(outpath_ccp_results + outfile + '_nmi.npy') or not os.path.exists(outpath_ccp_results + outfile + '_sil.npy'):
            LABELS, ARI, NMI, SIL = computeKMeans(X_ccp, y)
            np.save(outpath_ccp_results + outfile + '_labels.npy', LABELS)
            np.save(outpath_ccp_results + outfile + '_ari.npy', ARI)
            np.save(outpath_ccp_results + outfile + '_nmi.npy', NMI)
            np.save(outpath_ccp_results + outfile + '_sil.npy', SIL)

        

#PCA
outpath_pca = constructPCApath(data)
outpath_pca_results = constructPCApathResults(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile = '%s_pca_n%d_state%d'%(data, n_components, state)
        X_pca = np.load(outpath_pca + outfile + '.npy')
        
        if not os.path.exists(outpath_pca_results + outfile + '_labels.npy') or not os.path.exists(outpath_pca_results + outfile + '_ari.npy')\
            or not os.path.exists(outpath_pca_results + outfile + '_nmi.npy') or not os.path.exists(outpath_pca_results + outfile + '_sil.npy'):
            LABELS, ARI, NMI, SIL = computeKMeans(X_pca, y)
            np.save(outpath_pca_results + outfile + '_labels.npy', LABELS)
            np.save(outpath_pca_results + outfile + '_ari.npy', ARI)
            np.save(outpath_pca_results + outfile + '_nmi.npy', NMI)
            np.save(outpath_pca_results + outfile + '_sil.npy', SIL)