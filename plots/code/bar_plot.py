# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:32:29 2023

@author: Yuta
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
This code generates the ARI bar-plot Figure 4 and 5

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
    
    
data_vec = ['GSE67835', 'GSE82187', 'GSE84133human4', 'GSE84133mouse1']

size = 10
width = 0.25
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

fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(4) * 1.5
ax.bar(x, ccp_ari, width, label = 'CCP')
ax.bar(x+0.25, ccp_umap_ari, width, label = 'CCP-UMAP')
ax.bar(x+0.5, ccp_tsne_ari, width, label = 'CCP-tSNE')
ax.bar(x+0.75, umap_ari, width, label = 'UMAP')
ax.bar(x+1, tsne_ari, width, label = 'tSNE')



ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_xticks(x + 0.475, ['GSE67835', 'GSE82187', 'GSE84133 H', 'GSE84133 M'])
ax.set_ylim([0.4, 1])
legend_name = ['CCP', 'CCP UMAP', 'CCP t-SNE', 'UMAP', 't-SNE']
ax.legend(legend_name, handletextpad=0.1, ncol=2, loc = 'upper left', framealpha = 0)
plt.savefig('../figures/results2_ari.png',  bbox_inches='tight', dpi = 500)



#####
data_vec = ['GSE75748cell', 'GSE75748time', 'GSE57249', 'GSE94820']

size = 10
width = 0.25
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

fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(4) * 1.5
ax.bar(x, ccp_ari, width, label = 'CCP')
ax.bar(x+0.25, ccp_umap_ari, width, label = 'CCP UMAP')
ax.bar(x+0.5, ccp_tsne_ari, width, label = 'CCP t-SNE')
ax.bar(x+0.75, umap_ari, width, label = 'UMAP')
ax.bar(x+1, tsne_ari, width, label = 't-SNE')



ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_xticks(x + 0.475, ['GSE75748 cell', 'GSE75748 time', 'GSE57249', 'GSE94820'])
ax.set_ylim([0.4, 1])

legend_name = ['CCP', 'CCP UMAP', 'CCP t-SNE', 'UMAP', 't-SNE']
ax.legend(legend_name, handletextpad=0.1, ncol=2, loc = 'upper left', framealpha = 0)
plt.savefig('../figures/results1_ari.png',  bbox_inches='tight', dpi = 500)