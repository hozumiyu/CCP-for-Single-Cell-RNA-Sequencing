# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:13:00 2023

@author: yutah
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import os
import csv

def makeFolder(path):
    try:
        os.makedirs(path)
    except:
        return
    return

def computeACC(infile):
    Y = np.load(infile)
    kfold_index = Y[:, 0]
    y_true = Y[:, 1]
    y_pred = Y[:, 2]
    maxFold = int(np.max(kfold_index))
    bac_vec = []; acc_vec = []
    for idx in range(maxFold):
        index = np.where(kfold_index == idx)[0]
        
        bac = balanced_accuracy_score(y_true[index], y_pred[index])
        acc = accuracy_score(y_true[index], y_pred[index])
        bac_vec.append(bac); acc_vec.append(acc)
    return bac_vec, acc_vec

def computeNCAccuracy(data, n_components, max_random_state, max_kf_state,kfold,  method):
    inpath = './%s_results_pca/%s_kfold_pca/%s_%s_%dfold_pca/'%(method, method, data, method , kfold)
    bac_vec = []; acc_vec = []
    for random_state in range(1, max_random_state + 1):
        for kf_state in range(1, max_kf_state + 1):
            file = '%s_pca_nc%d_state%d_%dfold_kfstate%d_%s.npy'%(data, n_components, random_state, kfold, kf_state, method)
            bac_temp, acc_temp = computeACC(inpath + file)
            bac_vec.append(np.mean(bac_temp)); acc_vec.append(np.mean(acc_temp))
    return bac_vec, acc_vec
            
    
    
def writeCVS( data, max_random_state, max_kf_state, kfold, method):
    outpath = './results/%s_results/'%(data) ; makeFolder(outpath)
    
    outfile = '%s_pca_results_%dfold_%s.csv'%(data, kfold, method)

    header = ['n_components', 'ACC', 'BAC', 'ACC VAR', 'BAC VAR']
    if data == 'GSE45719' and kfold == 5:
        n_components_vec = [50,100,150,200]
    else:
        n_components_vec = [50,100,150,200,250,300]

    
    with open(outpath + outfile, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for n_components in n_components_vec:
            bac_vec, acc_vec = computeNCAccuracy(data, n_components, max_random_state, max_kf_state,kfold,  method)
            results = {'n_components': n_components, 'ACC': np.mean(acc_vec), 'BAC':np.mean(bac_vec), 'ACC VAR':np.var(acc_vec), 'BAC VAR':np.var(bac_vec) }
            writer.writerow(results)
    return
data_vec = ['GSE45719', 'GSE67835', 'GSE75748cell', 'GSE75748time', 'GSE82187', 'GSE84133human1', 'GSE84133human2', 'GSE84133human3', 'GSE84133human4', 'GSE84133mouse1', 'GSE84133mouse2', 'GSE89232', 'GSE94820', 'GSE59114']

data_vec =  ['GSE45719', 'GSE67835', 'GSE75748cell', 'GSE75748time', 'GSE82187', 'GSE84133mouse1', 'GSE84133mouse2', 'GSE89232', 'GSE94820', 'GSE59114']
method_vec = ['knn', 'svm', 'rf', 'gbdt']
max_random_state = 10
max_kf_state = 10
kfold_vec = [5, 10]
for data in data_vec:
    
    for method in method_vec:
        for kfold in kfold_vec:
            writeCVS(data, max_random_state, max_kf_state, kfold, method)