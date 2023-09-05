# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:34:23 2023

@author: yutah
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np

'''
Code associated with clustering and its scores
'''

def computeClusterScore(X, y, label):
    #compute the clustering scores
    #ARI, NMI and silhouette scores
    ari = adjusted_rand_score(y, label)
    nmi = normalized_mutual_info_score(y, label)
    sil = silhouette_score(X, label)
    return ari, nmi, sil
    

def computeKMeans(X, y, max_state = 30, scale = 'standard'):
    '''
        compute k-means clustering for the reduction with 30 random instance
        input:
            X: M x N data
            y: M * 1 true labels
            max_state: number of k-means state
        return:
            LABELS: max_state * M label from k-means
            ARI: max_state * 1 ari for each instance of k-means
            NMI: max_state * 1 nmi for each instance of k-means
            Sil: max_state * 1 silhouette score for each instance of k-means
    '''
    M = X.shape[0]
    n_clusters = np.unique(y).shape[0]
    X_scaled = StandardScaler().fit_transform(X)
    LABELS = np.zeros([max_state, M])
    ARI = np.zeros(max_state); NMI = np.zeros(max_state); SIL = np.zeros(max_state)
    for state in range(max_state):
        myKM = KMeans(n_clusters = n_clusters, random_state = state)
        myKM.fit(X_scaled)
        label = myKM.labels_
        ARI[state], NMI[state], SIL[state] = computeClusterScore(X, y, label)
        LABELS[state, :] = label
    return LABELS, ARI, NMI, SIL
    