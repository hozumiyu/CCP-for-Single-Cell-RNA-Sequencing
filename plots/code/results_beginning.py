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
This produces figure 1
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
        ax.scatter(X[index, 0], X[index, 1], color = color[idx], s =3 )
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def extractLegend(data, y):
    labels = np.unique(y)
    labels.sort()
    inpath = '../../data/%s/'%(data)
    file = open(inpath + '%s_full_labeldict.txt'%(data), 'r')
    lines = file.readlines()
    file.close()
    mapping = eval(lines[0])
    legend = []
    for l in labels:
        legend.append(mapping[l])
    return legend

def remapLabels(y, mapping):
    new_y = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        new_y[i] = mapping[y[i]]
    return new_y


fig, ax = plt.subplots(nrows=1, ncols = 2, figsize = ( 4.25, 2.125))
outpath = '../figures/'; makeFolder(outpath)
###########################################################################
#GSE75748cell
data, method, cutoff, n_components = 'GSE84133human3' , 'tsne','0.5','200'
y = load_y(data)
mapping = {0: 0, 1:7, 2:1, 3:2 ,4:3, 5:5, 6:6, 8:4, 11:8}
y = remapLabels(y, mapping)
legend = ['Acinar', r'$\alpha$', r'$\beta$', r'$\delta$', r'$\gamma$', 'Ductal', 'Endo', 'Activated ste', 'Quiescent ste']
#legend = ['Acinar', 'Alpha', 'Beta', 'Delta', 'Gamma', 'Ductal', 'Endothelial', 'Activated stel', 'Quiescent ste']


inpath = '../../plotting_features/%s_features/'%( data)
file = '%s_c%s_n%s_s1_%s.npy'%(data, cutoff, n_components, method)
X = np.load(inpath + file)
ax[0] = plot(X, y, ax[0])


file = '%s_%s_s1.npy'%(data, method)
X = np.load(inpath + file)
ax[1] = plot(X, y, ax[1])

'''
legend = ax[0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 2, 0.03),fancybox=False, shadow=False, ncol=5, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
#ax[0].set_ylabel('GSE75748 cell', rotation=90)
'''
###########################################################################


#fig.subplots_adjust(wspace=0)

ax[0].set_title('CCP assisted t-SNE')
ax[1].set_title('Unassisted t-SNE')
size = 10
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title
fig.savefig(outpath + 'results_beginning.png', bbox_inches='tight', dpi = 500)
