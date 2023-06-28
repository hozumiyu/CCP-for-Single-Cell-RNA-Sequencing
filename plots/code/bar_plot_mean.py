# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:32:29 2023

@author: Yuta
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
This produces the mean ari plot (Figure 6)

'''
def find_result(data, lines):
    col_mapping = {'50': 1, '100':2, '150':3, '200':4, '250':5, '300':6}
    row_mapping = {'0.5': 1, '0.6':2, '0.7':3, '0.8':4, '0.9':1}
    for idx, line in enumerate(lines):
        line = line.strip()
        if line == data:
            break
    ccp, ccp_cutoff, ccp_n_components = lines[idx+1].split(','); ccp_n_components = ccp_n_components.strip()
    umap, umap_cutoff, umap_n_components = lines[idx+2].split(','); umap_n_components = umap_n_components.strip()
    tsne, tsne_cutoff, tsne_n_components = lines[idx+3].split(','); tsne_n_components = tsne_n_components.strip()
    
    ccp_path = '../../results/%s_results/%s_ccp.csv'%(data, data)
    file = open(ccp_path)
    ccp_lines = file.readlines()
    file.close()
    ccp_temp = ccp_lines[row_mapping[ccp_cutoff]].split(',')
    ccp_ari = eval(ccp_temp[col_mapping[ccp_n_components]])
    
    umap_path = '../../results/%s_results/%s_ccp_umap.csv'%(data, data)
    file = open(umap_path)
    umap_lines = file.readlines()
    file.close()
    umap_temp = umap_lines[row_mapping[umap_cutoff]].split(',')
    ccp_umap_ari = eval(umap_temp[col_mapping[umap_n_components]])
    
    tsne_path = '../../results/%s_results/%s_ccp_tsne.csv'%(data, data)
    file = open(tsne_path)
    tsne_lines = file.readlines()
    file.close()
    tsne_temp = tsne_lines[row_mapping[tsne_cutoff]].split(',')
    ccp_tsne_ari = eval(tsne_temp[col_mapping[tsne_n_components]])
    
    
    file = open('../../results/%s_results/%s_umap_tsne.csv'%(data, data))
    umap_tsne_lines = file.readlines()
    file.close()
    umap_ari = umap_tsne_lines[1].split(','); umap_ari = eval(umap_ari[1])
    tsne_ari = umap_tsne_lines[2].split(','); tsne_ari = eval(tsne_ari[1])
    return ccp_ari, ccp_umap_ari, ccp_tsne_ari, umap_ari, tsne_ari 

    
data_vec = ['GSE67835', 'GSE82187', 'GSE84133human4', 'GSE84133mouse1', 'GSE75748cell', 'GSE75748time', 'GSE57249', 'GSE94820']


size = 10

plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title


file = open('parameters.txt')
lines = file.readlines()
file.close()


ccp_ari = []; ccp_umap_ari = []; ccp_tsne_ari = []; umap_ari = []; tsne_ari = [];
for data in data_vec:
    ccp_ari_temp, ccp_umap_ari_temp, ccp_tsne_ari_temp, umap_ari_temp, tsne_ari_temp = find_result(data, lines)
    ccp_ari.append(ccp_ari_temp); ccp_umap_ari.append(ccp_umap_ari_temp); ccp_tsne_ari.append(ccp_tsne_ari_temp)
    umap_ari.append(umap_ari_temp); tsne_ari.append(tsne_ari_temp)
ccp_ari = np.mean(ccp_ari)
ccp_umap_ari = np.mean(ccp_umap_ari)
ccp_tsne_ari = np.mean(ccp_tsne_ari)
umap_ari =  np.mean(umap_ari)
tsne_ari = np.mean(tsne_ari)


fig = plt.figure(figsize = (6,3))
ax = fig.add_subplot(111)
width = 0.125
diff = 0.15
x = 0
ax.bar(x, ccp_ari, width, label = 'CCP')
ax.bar(x+diff, ccp_umap_ari, width, label = 'CCP-UMAP')
ax.bar(x+diff*2, ccp_tsne_ari, width, label = 'CCP-tSNE')
ax.bar(x+diff*3, umap_ari, width, label = 'UMAP')
ax.bar(x+diff*4, tsne_ari, width, label = 'tSNE')


xx = [x, x + diff, x +  diff*2, x + diff*3, x + diff*4]
y = [ccp_ari.round(3), ccp_umap_ari.round(3), ccp_tsne_ari.round(3), 0.630,  tsne_ari.round(3)]
color = ['blue', 'orange', 'green', 'red', 'purple']
for i in range(len(xx)):
    if i  == 3:
        plt.text(xx[i], y[i]+0.006, '0.630', ha = 'center')
    else: 
        plt.text(xx[i], y[i]+0.006, y[i], ha = 'center')
ax.set_ylabel('Mean ARI')


ax.set_yticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80])
ax.set_ylim([0.5, 0.8])
ax.set_xticks( [x, x + diff, x +  diff*2, x + diff*3, x + diff*4] , 
              ['Mean\nCCP', 'Mean\nCCP UMAP', 'Mean\nCCP t-SNE', 'Mean\nUMAP', 'Mean\nt-SNE'])
#legend_name = ['CCP', 'CCP UMAP', 'CCP t-SNE', 'UMAP', 't-SNE']
#ax.legend(legend_name, handletextpad=0.1, ncol=2, loc = 'upper left', framealpha = 0)
plt.savefig('../figures/mean_ari.png',  bbox_inches='tight', dpi = 500)

