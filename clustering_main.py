# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:56:48 2023

@author: yutah
"""
import os, sys
import numpy as np
from codes.auxilary import load_X, load_y, preprocess_data
from codes.constructOutpath import constructCCPpath, constructPCApath, constructCCPpathResults, constructPCApathResults, constructNMFpath, constructNMFpathResults
from codes.clustering import computeKMeans

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
#n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
#max_state = 20
n_components_vec = [int(sys.argv[2])]

y = load_y(data)


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
            print('%s CCP n%d state%d: ARI: %.4f, NMI: %.4f, SIL: %.4f'%(data, n_components, state, np.mean(ARI), np.mean(NMI), np.mean(SIL)))
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
        infile = '%s_pca_state%d.npy'%(data, state)
        X_pca = np.load(outpath_pca + infile)[:, :n_components]
        
        if not os.path.exists(outpath_pca_results + outfile + '_labels.npy') or not os.path.exists(outpath_pca_results + outfile + '_ari.npy')\
            or not os.path.exists(outpath_pca_results + outfile + '_nmi.npy') or not os.path.exists(outpath_pca_results + outfile + '_sil.npy'):
            LABELS, ARI, NMI, SIL = computeKMeans(X_pca, y)
            print('%s PCA n%d state%d: ARI: %.4f, NMI: %.4f, SIL: %.4f'%(data, n_components, state, np.mean(ARI), np.mean(NMI), np.mean(SIL)))
            np.save(outpath_pca_results + outfile + '_labels.npy', LABELS)
            np.save(outpath_pca_results + outfile + '_ari.npy', ARI)
            np.save(outpath_pca_results + outfile + '_nmi.npy', NMI)
            np.save(outpath_pca_results + outfile + '_sil.npy', SIL)
            
#NMF
outpath_nmf = constructNMFpath(data)
outpath_nmf_results = constructNMFpathResults(data)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        outfile = '%s_nmf_n%d_state%d'%(data, n_components, state)
        infile = '%s_nmf_n%d_state%d.npy'%(data, n_components, state)
        X_nmf = np.load(outpath_nmf + infile)[:, :n_components]
        if not os.path.exists(outpath_nmf_results + outfile + '_labels.npy') or not os.path.exists(outpath_nmf_results + outfile + '_ari.npy')\
            or not os.path.exists(outpath_nmf_results + outfile + '_nmi.npy') or not os.path.exists(outpath_nmf_results + outfile + '_sil.npy'):
            LABELS, ARI, NMI, SIL = computeKMeans(X_nmf, y, scale = 'nmf')
            print('%s nmf  n%d: ARI: %.4f, NMI: %.4f, SIL: %.4f'%(data, n_components, np.mean(ARI), np.mean(NMI), np.mean(SIL)))
            np.save(outpath_nmf_results + outfile + '_labels.npy', LABELS)
            np.save(outpath_nmf_results + outfile + '_ari.npy', ARI)
            np.save(outpath_nmf_results + outfile + '_nmi.npy', NMI)
            np.save(outpath_nmf_results + outfile + '_sil.npy', SIL)