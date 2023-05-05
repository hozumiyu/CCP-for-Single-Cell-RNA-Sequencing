# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:07:16 2023

@author: yutah
"""

import os


def makeFolder(outpath):
    try: 
        os.makedirs(outpath)
    except:
        return
    return


def constructCCPpath(data):
    outpath = './features/%s_features/%s_ccp_features/'%(data, data); makeFolder(outpath)
    return outpath

def constructCCPpathResults(data):
    outpath = './raw_results/%s_rawResults/%s_ccp_rawResults_clustering/'%(data, data); makeFolder(outpath)
    return outpath

def constructCCPpathResultsClassification(data):
    outpath = './raw_results/%s_rawResults/%s_ccp_rawResults_classification/'%(data, data); makeFolder(outpath)
    return outpath

def constructCCPpathResultsRSI(data):
    outpath = './raw_results/%s_rawResults/%s_ccp_rawResults_rsi/'%(data, data); makeFolder(outpath)
    return outpath

def constructPCApath(data):
    outpath = './features/%s_features/%s_pca_features/'%(data, data); makeFolder(outpath)
    return outpath


def constructPCApathResults(data):
    outpath = './raw_results/%s_rawResults/%s_pca_rawResults_clustering/'%(data, data); makeFolder(outpath)
    return outpath

def constructPCApathResultsClassification(data):
    outpath = './raw_results/%s_rawResults/%s_pca_rawResults_classification/'%(data, data); makeFolder(outpath)
    return outpath


def constructPCApathResultsRSI(data):
    outpath = './raw_results/%s_rawResults/%s_pca_rawResults_rsi/'%(data, data); makeFolder(outpath)
    return outpath