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



data_vec = ['GSE45719', 'GSE67835', 'GSE75748cell', 'GSE75748time', 'GSE82187', 'GSE84133human1', 'GSE84133human2', 'GSE84133human3', 'GSE84133human4', 'GSE84133mouse1', 'GSE84133mouse2', 'GSE89232', 'GSE94820', 'GSE59114']

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



def mergeSTD(data):
    n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    inpath_ccp_results = './rawResults_test/%s_rawResults_test/'%(data); makeFolder(inpath_ccp_results)
    ari = []; nmi = []
    for n_components in n_components_vec:
        ari_nc = []; nmi_nc = []
        for state in range(1, 21):
            infile = '%s_ccp_n%d_state%d'%(data, n_components, state)
            ari_temp = np.load(inpath_ccp_results + infile + '_std_ari.npy'); ari_temp = np.mean(ari_temp)
            nmi_temp = np.load(inpath_ccp_results + infile + '_std_nmi.npy'); nmi_temp = np.mean(nmi_temp)
            ari_nc.append(ari_temp); nmi_nc.append(nmi_temp)
        ari_nc = np.mean(ari_nc); nmi_nc = np.mean(nmi_nc)
        ari.append(ari_nc); nmi.append(nmi_nc)
    return ari, nmi

def mergeRAW(data):
    n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    inpath_ccp_results = './rawResults_test/%s_rawResults_test/'%(data); makeFolder(inpath_ccp_results)
    ari = []; nmi = []
    for n_components in n_components_vec:
        ari_nc = []; nmi_nc = []
        for state in range(1, 21):
            infile = '%s_ccp_n%d_state%d'%(data, n_components, state)
            ari_temp = np.load(inpath_ccp_results + infile + '_raw_ari.npy'); ari_temp = np.mean(ari_temp)
            nmi_temp = np.load(inpath_ccp_results + infile + '_raw_nmi.npy'); nmi_temp = np.mean(nmi_temp)
            ari_nc.append(ari_temp); nmi_nc.append(nmi_temp)
        ari_nc = np.mean(ari_nc); nmi_nc = np.mean(nmi_nc)
        ari.append(ari_nc); nmi.append(nmi_nc)
    return ari, nmi

def mergeSUM(data):
    n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    inpath_ccp_results = './rawResults_test/%s_rawResults_test/'%(data); makeFolder(inpath_ccp_results)
    ari = []; nmi = []
    for n_components in n_components_vec:
        ari_nc = []; nmi_nc = []
        for state in range(1, 21):
            infile = '%s_ccp_n%d_state%d'%(data, n_components, state)
            ari_temp = np.load(inpath_ccp_results + infile + '_sum_ari.npy'); ari_temp = np.mean(ari_temp)
            nmi_temp = np.load(inpath_ccp_results + infile + '_sum_nmi.npy'); nmi_temp = np.mean(nmi_temp)
            ari_nc.append(ari_temp); nmi_nc.append(nmi_temp)
        ari_nc = np.mean(ari_nc); nmi_nc = np.mean(nmi_nc)
        ari.append(ari_nc); nmi.append(nmi_nc)
    return ari, nmi
        
def writeCSV(data, ari_raw, nmi_raw, ari_sum, nmi_sum, ari_std, nmi_std):
    n_components_vec = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    outpath = './results_test/'; makeFolder(outpath)
    outfile = '%s_ccp_test.csv'%(data)
    file = open(outpath + outfile , 'w')
    lines = 'n_components, ari raw, ari sum,  ari std, nmi raw, nmi sum, nmi std \n'
    file.writelines(lines)
    for idx, nc in enumerate(n_components_vec):
        line = ','.join([str(nc), str(ari_raw[idx]), str(ari_sum[idx]), str(ari_std[idx]), str(nmi_raw[idx]), str(nmi_sum[idx]),str(nmi_std[idx])]) + '\n'
        file.writelines(line)
    file.close()
    
        
    
    return

#CCP 
for data in data_vec:
    ari_raw, nmi_raw = mergeRAW(data)
    ari_sum, nmi_sum = mergeSUM(data)
    ari_std, nmi_std = mergeSTD(data)
    writeCSV(data, ari_raw, nmi_raw, ari_sum, nmi_sum, ari_std, nmi_std)