# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:25:41 2023

@author: yutah
"""

import os
from codes.constructOutpath import constructCCPpath, constructPCApath
from codes.constructOutpath import constructCCPpathResults, constructPCApathResults
from codes.constructOutpath import constructCCPpathResultsRSI, constructPCApathResultsRSI
from codes.computeRSI import computeRSIclustering, computeRSItrue, computeRSI5fold
from codes.auxilary import load_y
import numpy as np

data = 'GSE75748time'
n_components_vec = [300]
max_state = 1

y = load_y(data)
unique_labels = np.unique(y)
unique_labels.sort()

#import data
outpath_ccp = constructCCPpath(data)
outpath_ccp_labels = constructCCPpathResults(data)
outpath_ccp_rsi = constructCCPpathResultsRSI(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile = '%s_ccp_n%d_state%d'%(data, n_components, state)
        X_ccp = np.load(outpath_ccp + outfile + '.npy')
        
        if not os.path.exists(outpath_ccp_rsi + outfile + '_clustering_rsscore.npy') or not os.path.exists(outpath_ccp_rsi + outfile + '_clustering_rsi.npy'):
            LABELS = np.load(outpath_ccp_labels + outfile + '_labels.npy')
            RS_SCORE, RSI = computeRSIclustering(X_ccp, LABELS, unique_labels)
            np.save(outpath_ccp_rsi + outfile + '_clustering_rsscore.npy', RS_SCORE)
            np.save(outpath_ccp_rsi + outfile + '_clustering_rsi.npy', RSI)
            
        if not os.path.exists(outpath_ccp_rsi + outfile + '_true_rsscore.npy') or not os.path.exists(outpath_ccp_rsi + outfile + '_true_rsi.npy'):
            RS_SCORE, RSI = computeRSItrue(X_ccp, y, unique_labels)
            np.save(outpath_ccp_rsi + outfile + '_true_rsscore.npy', RS_SCORE)
            np.save(outpath_ccp_rsi + outfile + '_true_rsi.npy', RSI)
        
        if not os.path.exists(outpath_ccp_rsi + outfile + '_5fold_rsscore.npy') or not os.path.exists(outpath_ccp_rsi + outfile + '_5fold_rsi.npy'):
            RS_SCORE, RSI = computeRSI5fold(X_ccp, X_ccp, y, unique_labels)
            np.save(outpath_ccp_rsi + outfile + '_5fold_rsscore.npy', RS_SCORE)
            np.save(outpath_ccp_rsi + outfile + '_5fold_rsi.npy', RSI)

outpath_pca = constructPCApath(data)
outpath_pca_labels = constructPCApathResults(data)
outpath_pca_rsi = constructPCApathResultsRSI(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile = '%s_pca_n%d_state%d'%(data, n_components, state)
        X_pca = np.load(outpath_pca + outfile + '.npy')
        
        if not os.path.exists(outpath_pca_rsi + outfile + '_clustering_rsscore.npy') or not os.path.exists(outpath_pca_rsi + outfile + '_clustering_rsi.npy'):
            LABELS = np.load(outpath_pca_labels + outfile + '_labels.npy')
            RS_SCORE, RSI = computeRSIclustering(X_pca, LABELS, unique_labels)
            np.save(outpath_pca_rsi + outfile + '_clustering_rsscore.npy', RS_SCORE)
            np.save(outpath_pca_rsi + outfile + '_clustering_rsi.npy', RSI)
            
        if not os.path.exists(outpath_pca_rsi + outfile + '_true_rsscore.npy') or not os.path.exists(outpath_pca_rsi + outfile + '_true_rsi.npy'):
            RS_SCORE, RSI = computeRSItrue(X_pca, y, unique_labels)
            np.save(outpath_pca_rsi + outfile + '_true_rsscore.npy', RS_SCORE)
            np.save(outpath_pca_rsi + outfile + '_true_rsi.npy', RSI)
        
        if not os.path.exists(outpath_pca_rsi + outfile + '_5fold_rsscore.npy') or not os.path.exists(outpath_pca_rsi + outfile + '_5fold_rsi.npy'):
            X_ccp = np.load(outpath_ccp + '%s_ccp_n%d_state%d.npy'%(data, n_components, state))
            RS_SCORE, RSI = computeRSI5fold(X_ccp, X_pca, y, unique_labels)
            np.save(outpath_pca_rsi + outfile + '_5fold_rsscore.npy', RS_SCORE)
            np.save(outpath_pca_rsi + outfile + '_5fold_rsi.npy', RSI)
        