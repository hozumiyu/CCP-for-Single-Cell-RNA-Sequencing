#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:53:28 2023

@author: yutaho
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
This makes figure 7
'''

def makeFolder(path):
    try: 
        os.makedirs(path)
    except:
        return    
    return 

def load_y(data):
    inpath = '../../data/%s/'%(data)
    y = pd.read_csv(inpath + '%s_full_labels.csv'%(data))
    y = np.array(list(y['Label'])).astype(int)
    y = drop_sample(y)
    return y

def drop_sample(y):
    labels = np.unique(y)
    good_index = []
    for l in labels:
        index = np.where(y == l)[0]
        if index.shape[0] > 15:
            good_index.append(index)
        else:
            print('label %d removed'%(l))
    good_index = np.concatenate(good_index)
    good_index.sort()
    
    return y[good_index]

def plot(X, y, ax):
    color = [plt.cm.tab10(l) for l in range(10)]
    color[7] = color[-1]
    labels = np.unique(y)
    labels.sort()
    for idx, l in enumerate(labels):
        index = np.where(y == l)[0]
        ax.scatter(X[index, 0], X[index, 1], color = color[idx], s =1 )
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

data_vec = []
data = 'GSE75748time'
legend_text = ['0hr', '12hr', '24hr', '36hr', '72hr', '96hr']
n_components_vec = [50, 100, 150, 200, 250, 300]
cutoff_ratio = [0.6, 0.7, 0.8, 0.9]
y = load_y(data)
inpath = '../../features/%s_features/%s_ccp_umap/'%(data, data)

fig, ax = plt.subplots(nrows=4, ncols = 6, figsize = (8.5, 5.6))
for col, n_components in enumerate(n_components_vec):
    for row, cutoff in enumerate(cutoff_ratio):
        file = '%s_ccp_n%d_c%.1f_umap.npy'%(data, n_components, cutoff)
        outpath = '../figures/'; makeFolder(outpath)
        y = load_y(data)
        X = np.load(inpath + file)
                
        ax[row, col] = plot(X, y, ax[row, col])
        ax[row, 0].set_ylabel('$v_{c}$ = ' + '%.1f'%(cutoff), rotation=90)
    ax[0, col].set_title('$N = $' + '%d'%(n_components))
    
legend = ax[row, 0].legend(legend_text, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]

fig.subplots_adjust(wspace=0, hspace = 0)
size = 10
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title
fig.savefig(outpath + '%s_cutoff_umap.png'%(data), bbox_inches='tight', dpi = 500)



inpath = '../../features/%s_features/%s_ccp_tsne/'%(data, data)

fig, ax = plt.subplots(nrows=4, ncols = 6, figsize = (8.5, 5.6))
for col, n_components in enumerate(n_components_vec):
    for row, cutoff in enumerate(cutoff_ratio):
        file = '%s_ccp_n%d_c%.1f_tsne.npy'%(data, n_components, cutoff)
        outpath = '../figures/'; makeFolder(outpath)
        y = load_y(data)
        X = np.load(inpath + file)
                
        ax[row, col] = plot(X, y, ax[row, col])
        ax[row, 0].set_ylabel('$v_{c}$ = ' + '%.1f'%(cutoff), rotation=90)
    ax[0, col].set_title('$N = $' + '%d'%(n_components))
    
legend = ax[row, 0].legend(legend_text, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
    
fig.subplots_adjust(wspace=0, hspace = 0)

size = 10
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title
fig.savefig(outpath + '%s_cutoff_tsne.png'%(data), bbox_inches='tight', dpi = 500)