# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:56:48 2023

@author: yutah
"""
import os, sys
import numpy as np
from codes.auxilary import load_X, load_y, preprocess_data
from codes.constructOutpath import constructCCPpath
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
def makeFolder(outpath):
    try:
        os.makedirs(outpath)
    except:
        return
    return

def KM_raw(X_ccp, y, outfile):
    ari = np.zeros(30)
    nmi = np.zeros(30)
    sil = np.zeros(30)
    labels = np.zeros([30, X_ccp.shape[0]])
    for idx in range(30):
        myKM = KMeans(n_clusters = np.unique(y).shape[0], random_state = idx)
        myKM.fit(X_ccp)
        l = myKM.labels_
        ari[idx] = adjusted_rand_score(y, l)
        nmi[idx] = normalized_mutual_info_score(y, l)
        sil[idx] = silhouette_score(X_ccp, l)
        labels[idx, :] = l
    np.save(outfile + '_raw_labels.npy', labels)
    np.save(outfile + '_raw_ari.npy', ari)
    np.save(outfile + '_raw_nmi.npy', nmi)
    np.save(outfile + '_raw_sil.npy', sil)
    print('raw', np.mean(ari))
    
    
def KM_standard(X_ccp, y, outfile):
    X_ccp_scaled = StandardScaler().fit_transform(X_ccp)
    ari = np.zeros(30)
    nmi = np.zeros(30)
    sil = np.zeros(30)
    labels = np.zeros([30, X_ccp.shape[0]])
    for idx in range(30):
        myKM = KMeans(n_clusters = np.unique(y).shape[0], random_state = idx)
        myKM.fit(X_ccp_scaled)
        l = myKM.labels_
        ari[idx] = adjusted_rand_score(y, l)
        nmi[idx] = normalized_mutual_info_score(y, l)
        sil[idx] = silhouette_score(X_ccp_scaled, l)
        labels[idx, :] = l
    np.save(outfile + '_std_labels.npy', labels)
    np.save(outfile + '_std_ari.npy', ari)
    np.save(outfile + '_std_nmi.npy', nmi)
    np.save(outfile + '_std_sil.npy', sil)
    print('std', np.mean(ari))
    
def KM_sum(X_ccp, y, outfile):
    X_ccp_scaled = X_ccp / (np.sum(X_ccp, axis = 1)[:, None])
    ari = np.zeros(30)
    nmi = np.zeros(30)
    sil = np.zeros(30)
    labels = np.zeros([30, X_ccp.shape[0]])
    for idx in range(30):
        myKM = KMeans(n_clusters = np.unique(y).shape[0], random_state = idx)
        myKM.fit(X_ccp_scaled)
        l = myKM.labels_
        ari[idx] = adjusted_rand_score(y, l)
        nmi[idx] = normalized_mutual_info_score(y, l)
        sil[idx] = silhouette_score(X_ccp_scaled, l)
        labels[idx, :] = l
    np.save(outfile + '_sum_labels.npy', labels)
    np.save(outfile + '_sum_ari.npy', ari)
    np.save(outfile + '_sum_nmi.npy', nmi)
    np.save(outfile + '_sum_sil.npy', sil)
    print('sum', np.mean(ari))

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
n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
#max_state = 20
n_components_vec = [int(sys.argv[2])]
y = load_y(data)




#CCP
outpath_ccp = constructCCPpath(data)
outpath_ccp_results = './rawResults_test/%s_rawResults_test/'%(data); makeFolder(outpath_ccp_results)
for n_components in n_components_vec:
    for state in range(1, max_state + 1):
        infile = '%s_ccp_n%d_state%d'%(data, n_components, state)
        X_ccp = np.load(outpath_ccp + infile + '.npy')
        
        outfile = outpath_ccp_results + infile
        KM_raw(X_ccp, y, outfile)
        KM_standard(X_ccp, y, outfile)
        KM_sum(X_ccp, y, outfile)