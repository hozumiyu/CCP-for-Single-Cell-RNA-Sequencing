# -*- coding: utf-8 -*-
"""
Created on Fri May  5 20:09:05 2023

@author: yutah
"""

import os
import numpy as np
from codes.constructOutpath import constructCCPpathResults, constructPCApathResults, constructNMFpathResults
from codes.constructOutpath import constructCCPpathResultsClassification, constructPCApathResultsClassification, constructNMFpathResultsClassification
from codes.constructOutpath import constructCCPpathResultsRSI, constructPCApathResultsRSI
from codes.computeRSI import computeRSIclustering, computeRSItrue, computeRSI5fold


def makeFolder(outpath):
    try:
        os.makedirs(outpath)
    except:
        return
    return

def writeCSV(outpath, lines):
    file = open(outpath, 'w')
    for line in lines:
        outline = ''
        for l in line:
            outline = outline +  str(l) + ','
        outline = outline[:-1] + '\n'
        file.write(outline)
    file.close()
    return
    


data_vec = ['GSE45719', 'GSE67835', 'GSE75748cell', 'GSE75748time', 'GSE82187', 'GSE84133human1', 'GSE84133human2', 'GSE84133human3', 'GSE84133human4', 'GSE84133mouse1', 'GSE84133mouse2', 'GSE89232', 'GSE94820']

max_state = 20
n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]


#CCP
for data in data_vec:
    outpath = './results/%s_results/'%(data); makeFolder(outpath)
    outpath_ccp_results = constructCCPpathResultsRSI(data)
    outfile = outpath + '%s_ccp_rsi.csv'%(data)
    lines = [['n_components,R clustering,S clustering,RS clustering,R 5fold,S 5fold,RS 5fold,R true,S true,RS true']]
    for n_components in n_components_vec:
        r_clustering_nc = []; s_clustering_nc = []; rs_clustering_nc = []
        r_5fold_nc = []; s_5fold_nc = []; rs_5fold_nc = []
        r_true_nc = []; s_true_nc = []; rs_true_nc = []
        for state in range(1, max_state + 1):
            infile = '%s_ccp_n%d_state%d'%(data, n_components, state)
            rsi_clustering = np.load(outpath_ccp_results + infile + '_clustering_rsi.npy'); 
            rsi_clustering = np.mean(rsi_clustering, axis = 1)
            rsi_clustering = np.mean(rsi_clustering, axis = 0)
            r, s, rs = rsi_clustering
            r_clustering_nc.append(r); s_clustering_nc.append(s); rs_clustering_nc.append(rs)
            
            rsi_5fold = np.load(outpath_ccp_results + infile + '_5fold_rsi.npy'); 
            rsi_5fold = np.mean(rsi_5fold, axis = 1)
            rsi_5fold = np.mean(rsi_5fold, axis = 0)
            r, s, rs = rsi_5fold
            r_5fold_nc.append(r); s_5fold_nc.append(s); rs_5fold_nc.append(rs)
            
            rsi_true = np.load(outpath_ccp_results + infile + '_true_rsi.npy'); 
            rsi_true = np.mean(rsi_true, axis = 1)
            rsi_true = np.mean(rsi_true, axis = 0)
            r, s, rs = rsi_5fold
            r_true_nc.append(r); s_true_nc.append(s); rs_true_nc.append(rs)
        line = [n_components, np.mean(r_clustering_nc), np.mean(s_clustering_nc), np.mean(rs_clustering_nc), 
                np.mean(r_5fold_nc), np.mean(s_5fold_nc), np.mean(rs_5fold_nc), 
                np.mean(r_true_nc), np.mean(s_true_nc), np.mean(rs_true_nc)]
        print(line)
        lines.append(line)
    writeCSV(outfile, lines)

#PCA
for data in data_vec:
    outpath = './results/%s_results/'%(data); makeFolder(outpath)
    outpath_pca_results = constructPCApathResultsRSI(data)
    outfile = outpath + '%s_pca_rsi.csv'%(data)
    lines = [['n_components,R clustering,S clustering,RS clustering,R 5fold,S 5fold,RS 5fold,R true,S true,RS true']]
    for n_components in n_components_vec:
        r_clustering_nc = []; s_clustering_nc = []; rs_clustering_nc = []
        r_5fold_nc = []; s_5fold_nc = []; rs_5fold_nc = []
        for state in range(1, max_state + 1):
            infile = '%s_pca_n%d_state%d'%(data, n_components, state)
            rsi_clustering = np.load(outpath_pca_results + infile + '_clustering_rsi.npy'); 
            rsi_clustering = np.mean(rsi_clustering, axis = 1)
            rsi_clustering = np.mean(rsi_clustering, axis = 0)
            r, s, rs = rsi_clustering
            r_clustering_nc.append(r); s_clustering_nc.append(s); rs_clustering_nc.append(rs)
            
            rsi_5fold = np.load(outpath_pca_results + infile + '_5fold_rsi.npy'); 
            rsi_5fold = np.mean(rsi_5fold, axis = 1)
            rsi_5fold = np.mean(rsi_5fold, axis = 0)
            r, s, rs = rsi_5fold
            r_5fold_nc.append(r); s_5fold_nc.append(s); rs_5fold_nc.append(rs)
            
            rsi_true = np.load(outpath_pca_results + infile + '_true_rsi.npy'); 
            rsi_true = np.mean(rsi_true, axis = 1)
            rsi_true = np.mean(rsi_true, axis = 0)
            r, s, rs = rsi_5fold
            r_true_nc.append(r); s_true_nc.append(s); rs_true_nc.append(rs)
        line = [n_components, np.mean(r_clustering_nc), np.mean(s_clustering_nc), np.mean(rs_clustering_nc), 
                np.mean(r_5fold_nc), np.mean(s_5fold_nc), np.mean(rs_5fold_nc), 
                np.mean(r_true_nc), np.mean(s_true_nc), np.mean(rs_true_nc)]
        print(line)
        lines.append(line)
    writeCSV(outfile, lines)