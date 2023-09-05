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
    outpath_ccp_results = constructCCPpathResultsClassification(data)
    outfile = outpath + '%s_ccp_classification.csv'%(data)
    lines = [['n_components,BA']]
    for n_components in n_components_vec:
        ba_nc = []; 
        for state in range(1, max_state + 1):
            infile = '%s_ccp_n%d_state%d'%(data, n_components, state)
            ba = np.load(outpath_ccp_results + infile + '_ba.npy'); ba = np.mean(ba)
            ba_nc.append(ba)
        line = [n_components, np.mean(ba_nc)]
        print(line)
        lines.append(line)
    writeCSV(outfile, lines)

#PCA
for data in data_vec:
    outpath = './results/%s_results/'%(data); makeFolder(outpath)
    outpath_pca_results = constructPCApathResultsClassification(data)
    outfile = outpath + '%s_pca_classification.csv'%(data)
    lines = [['n_components,BA']]
    for n_components in n_components_vec:
        ari_nc = []; nmi_nc = []; sil_nc = []
        for state in range(1, max_state + 1):
            infile = '%s_pca_n%d_state%d'%(data, n_components, state)
            ba = np.load(outpath_pca_results + infile + '_ba.npy'); ba = np.mean(ba)
            ba_nc.append(ba)
        line = [n_components, np.mean(ba_nc)]
        print(line)
        lines.append(line)
    writeCSV(outfile, lines)
