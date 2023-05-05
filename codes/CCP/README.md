# CCP
Correlated Clustering and Projection for Dimensionality Reduction

CCP is a data-domain dimensionality reduction algorithm, and has the following 2 steps.
1) The features are clustered, according to their correlation/similarity
2) The features clusters are projected using flexiblity rigidity index into 1 descriptor


Full details of the methodology and theoretical motivation can be found on https://arxiv.org/abs/2206.04189

A simple example is performed using the TCGA-PANCAN data, which can be downloaded from UCI repository https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq.
1) Download the data and extract the file into toy_data
2) run the command example.py

Please use the following citation:
@article{hozumi2022ccp,

  title={Ccp: Correlated clustering and projection for dimensionality reduction},
  
  author={Hozumi, Yuta and Wang, Rui and Wei, Guo-Wei},
  
  journal={arXiv preprint arXiv:2206.04189},
  
  year={2022}
}
