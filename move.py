# -*- coding: utf-8 -*-
"""
Created on Fri May  5 20:09:05 2023

@author: yutah
"""

import os
import shutil

def makeFolder(outpath):
    try:
        os.makedirs(outpath)
    except:
        return
    return


data_vec = ['GSE45719', 'GSE67835', 'GSE75748cell', 'GSE75748time', 'GSE82187', 'GSE84133human1', 'GSE84133human2', 'GSE84133human3', 'GSE84133human4', 'GSE84133mouse1', 'GSE84133mouse2', 'GSE89232', 'GSE94820', 'GSE59114']

'''
for data in data_vec:
    inpath = './features/%s_features/%s_ccp_features/'%(data, data)
    for state in range(1, 21):
        outpath = './features2/%s_features/%s_ccp_features/%s_ccp_features_state%d/'%(data, data, data, state); makeFolder(outpath)
        outzip =  './features2/%s_features/%s_ccp_features/%s_ccp_features_state%d.zip'%(data, data, data, state)
        for nc in [50, 100, 150, 200, 250, 300]:
            file = '%s_ccp_n%d_state%d.npy'%(data, nc, state)
            shutil.copyfile(inpath + file, outpath + file)
        shutil.make_archive(outzip, 'zip', outpath)
        shutil.rmtree(outpath) 
        
'''

for data in data_vec:
    inpath = './features3/%s_features/%s_pca_features/'%(data, data)
    for state in range(1, 21):
        outpath = './features/%s_features/%s_pca_features/'%(data, data); makeFolder(outpath)
        file = '%s_pca_state%d.npy'%(data, state)
        shutil.copyfile(inpath + file, outpath + file)