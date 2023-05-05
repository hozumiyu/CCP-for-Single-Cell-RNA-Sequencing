# CCP-for-Single-Cell-RNA-Sequencing

**OVERVIEW**
In this repository, we utilize Correlated Clustering and Projection (CCP) to reduce single cell RNA sequencing (scRNA-seq) data.
To successfully reproduce the results, please run the codes in the following order.
1) reduction_main.py
2) clustering_main.py
3) classification_main.py
4) rs_main.py

reduction_main.py compute dimensionality reduction of the scRNA-seq using CCP and PCA
clustering_main.py compute the k-means clustering using the reduced data
classification_main.py computes the support vector machine classifier using the reduced data
rs_main.py compute the R-, S- RS-scores and the RSI for clustering, 5-fold cross-validation and the true data

**CCP**
The original paper of CCP can be found at https://arxiv.org/abs/2206.04189
We have a web-based service at https://weilab.math.msu.edu/CCP/
The source code is implemented in python and is available at https://github.com/hozumiyu/CCP
For this repository scale = 6.0, power = 2.0, ktype = 'exp' was utilized

**RSI**
The source code can be found at https://github.com/hozumiyu/RSI

**Data Availability**
The data is available from Gene Omnibus website
The detailed instructions on what to download and how to process can be found at https://github.com/hozumiyu/SingleCellDataProcess

