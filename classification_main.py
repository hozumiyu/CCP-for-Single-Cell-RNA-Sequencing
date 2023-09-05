# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:56:48 2023

@author: yutah
"""
import os,sys
import numpy as np
from codes.auxilary import load_X, load_y, preprocess_data
from codes.constructOutpath import constructCCPpath, constructPCApath, constructCCPpathResultsClassification, constructPCApathResultsClassification, constructNMFpath, constructNMFpathResultsClassification 
from codes.classification import compute5foldClassification

data = sys.argv[1]
max_state = 20
#data = 'GSE45719'
#data = 'GSE67835'
#data = 'GSE75748cell'
#data = 'GSE75748time'
#data = 'GSE82187'
#data = 'GSE84133human1'
#data = 'GSE84133human2'
#data = 'GSE84133human3'
#data = 'GSE84133human4'
#data = 'GSE84133mouse1'
#data = 'GSE84133mouse2'
#data = 'GSE89232'
#data = 'GSE94820'
#data = 'GSE59114'
n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
n_components_vec = [int(sys.argv[2])]

y = load_y(data)


#CCP
outpath_ccp = constructCCPpath(data)
outpath_ccp_results = constructCCPpathResultsClassification(data)
outpath_pca = constructPCApath(data)
outpath_pca_results = constructPCApathResultsClassification(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile_ccp = '%s_ccp_n%d_state%d'%(data, n_components, state); outfile_pca = '%s_pca_n%d_state%d'%(data, n_components, state)
        X_ccp = np.load(outpath_ccp + outfile_ccp + '.npy')
        infile = '%s_pca_state%d.npy'%(data, state)
        X_pca = np.load(outpath_pca + infile)[:, :n_components]
        if not os.path.exists(outpath_ccp_results + outfile_ccp + '_ba.npy') or not os.path.exists(outpath_pca_results + outfile_pca + '_ba.npy'):
            BA_ccp, BA_pca = compute5foldClassification(X_ccp, X_pca, y)
            print('%s n%d state%d: CCP BA: %.4f, PCA BA:%.4f'%(data, n_components, state, np.mean(BA_ccp), np.mean(BA_pca)))
            np.save(outpath_ccp_results + outfile_ccp + '_ba.npy', BA_ccp)
            np.save(outpath_pca_results + outfile_pca + '_ba.npy', BA_pca)


outpath_ccp = constructCCPpath(data)
outpath_ccp_results = constructCCPpathResultsClassification(data)
outpath_nmf = constructNMFpath(data)
outpath_nmf_results = constructNMFpathResultsClassification(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile_ccp = '%s_ccp_n%d_state%d'%(data, n_components, state); outfile_nmf = '%s_nmf_n%d_state%d'%(data, n_components, state)
        X_ccp = np.load(outpath_ccp + outfile_ccp + '.npy')
        infile = '%s_nmf_n%d_state%d.npy'%(data, n_components, state)
        X_nmf = np.load(outpath_nmf + infile)
        if not os.path.exists(outpath_ccp_results + outfile_ccp + '_ba.npy') or not os.path.exists(outpath_nmf_results + outfile_nmf + '_ba.npy'):
            BA_ccp, BA_pca = compute5foldClassification(X_ccp, X_nmf, y)
            #print('%s n%d state%d: CCP BA: %.4f, PCA BA:%.4f'%(data, n_components, state, np.mean(BA_ccp), np.mean(BA_pca)))
            #np.save(outpath_ccp_results + outfile_ccp + '_ba.npy', BA_ccp)
            np.save(outpath_nmf_results + outfile_nmf + '_ba.npy', BA_pca)

