# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:51:42 2016

@author: jonyoung
"""

import numpy as np
from sklearn import svm, cross_validation, metrics, manifold
import connectivity_utils as utils
import pandas as pd
import csv
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

# define distance functions
# Log-Euclidean metric based distance
def dLogE(C_1, C_2):
    
    # set value of regulatisation parameter gamma
    gamma = 1.0    
    
    # remove negative connectivities
    C_1[C_1 < 0] = 0;
    C_2[C_2 < 0] = 0;  
    
    # reshape matrices and calculate Laplacians
    M1 = np.reshape(C_1, (90, 90))
    M2 = np.reshape(C_2, (90, 90))
    D1 = np.diag(np.sum(M1, axis=1))
    D2 = np.diag(np.sum(M2, axis=1))
    L1 = D1 - M1
    L2 = D2 - M2
    S1 = L1 + (gamma * np.eye(90))
    S2 = L2 + (gamma * np.eye(90))
    return la.norm((la.logm(S1) - la.logm(S2)), ord='fro')

# DISTANCE between two covariance matrices according to eq'n 2 in Barachant,
# Alexandre, and Marco Congedo. "A Plug & Play P300 BCI Using Information 
# Geometry." arXiv preprint arXiv:1409.0107 (2014).
def dIG(C_1, C_2) :
    
    M_1 = np.reshape(C_1, (90, 90))
    M_2 = np.reshape(C_2, (90, 90))    
    M_1_pow = la.fractional_matrix_power(M_1, -0.5)
    return la.norm(la.logm(np.linalg.multi_dot([M_1_pow, M_2, M_1_pow])), ord='fro')

# set directories
dataset_1_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/'
dataset_3_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL3_timecourse/'

# import and process sparse inverse covariance data and metadata from dataset 1
# import dataset 1 sparse inverse covariances
dataset_1_cov_data = np.loadtxt(dataset_1_dir + 'sparse_inverse_covariance_data.csv', delimiter=',')

# import dataset 1 sparse inverse covariance files 
dataset_1_cov_files = pd.read_csv(dataset_1_dir + 'sparse_inverse_covariance_files.csv').T

# put these in a df
dataset_1_cov = pd.DataFrame(data=dataset_1_cov_data)
dataset_1_cov['file'] = dataset_1_cov_files.index

# convert format of file name so they can be matched
dataset_1_cov['file'] = dataset_1_cov['file'].apply(lambda x: x.split('/')[-1].split('_')[-1].zfill(7))

# import and process full dataset 1 files list and metadata to get labels
# import original dataset 1 data, files
dataset_1_data, dataset_1_files = utils.load_connectivity_data('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/')

# import dataset 1 labels
dataset_1_labels = utils.load_labels('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/')

# put labels alongside files in a DF
dataset_1_metadata = pd.DataFrame(columns=['file', 'label'])
dataset_1_metadata['file'] = dataset_1_files
dataset_1_metadata['label'] = dataset_1_labels 

# convert format of file name so they can be matched
dataset_1_metadata['file'] = dataset_1_metadata['file'].apply(lambda x: x.split('/')[-1].split('_')[-1])

# join the DFs to match labels with spare inverse cov data
dataset_1_cov = dataset_1_cov.merge(dataset_1_metadata, how='inner', on='file')

# extract the data and labels
dataset_1_cov_data = dataset_1_cov.iloc[:,0:8100].as_matrix()
dataset_1_cov_labels = np.array(dataset_1_cov['label'].tolist())

# shuffle the rows and labels
r = np.random.permutation(len(dataset_1_cov_labels))
dataset_1_cov_data_s = dataset_1_cov_data[r, :]
dataset_1_cov_labels_s = dataset_1_cov_labels[r]

# import and process sparse inverse covariance data and metadata from dataset 3
# import dataset 1 sparse inverse covariances and files
dataset_3_cov_data = np.loadtxt(dataset_3_dir + 'sparse_inverse_covariance_data.txt', delimiter=',')
with open(dataset_3_dir + 'sparse_inverse_covariance_files.csv', 'rb') as f:
    reader = csv.reader(f)
    dataset_3_cov_files = list(reader)[0]

# generate list of labels from file names: 1=pat (patient), 0=con (control)
dataset_3_cov_labels = np.array([1 if 'pat' in filename else 0 for filename in dataset_3_cov_files])

# shuffle the rows and labels
r = np.random.permutation(len(dataset_3_cov_labels))
dataset_3_cov_data_s = dataset_3_cov_data[r, :]
dataset_3_cov_labels_s = dataset_3_cov_labels[r]

#calculate distance matrices seperately for the datasets
#D_1 = squareform(pdist(dataset_1_cov_data_s, dLogE))
#
## do t-SNE embedding
#model = manifold.TSNE(n_components = 2, metric='precomputed')
#D_1_trans = model.fit_transform(D_1)
#plt.scatter(D_1_trans[dataset_1_cov_labels_s == 0, 0], D_1_trans[dataset_1_cov_labels_s == 0, 1], color='blue')   
#plt.scatter(D_1_trans[dataset_1_cov_labels_s == 1, 0], D_1_trans[dataset_1_cov_labels_s == 1, 1], color='red')
#
#clf = svm.SVC(kernel='linear')
#scores = cross_validation.cross_val_score(clf, D_1_trans, dataset_1_cov_labels_s, cv=10)
#print scores 
dataset_1_mean = np.reshape(np.mean(dataset_1_cov_data, axis = 0), (90, 90))
dataset_3_mean = np.reshape(np.mean(dataset_3_cov_data, axis = 0), (90, 90))
dataset_1_pos_mean = np.reshape(np.mean(dataset_1_cov_data[dataset_1_cov_labels == 1, :], axis = 0), (90, 90))
dataset_3_pos_mean = np.reshape(np.mean(dataset_3_cov_data[dataset_3_cov_labels == 1, :], axis = 0), (90, 90))
dataset_1_neg_mean = np.reshape(np.mean(dataset_1_cov_data[dataset_1_cov_labels == 0, :], axis = 0), (90, 90))
dataset_3_neg_mean = np.reshape(np.mean(dataset_3_cov_data[dataset_3_cov_labels == 0, :], axis = 0), (90, 90))

# distance between means
dist_means = la.norm(la.logm(dataset_1_mean) - la.logm(dataset_3_mean), 'fro')
dist_means_pos = la.norm(la.logm(dataset_1_pos_mean) - la.logm(dataset_3_pos_mean), 'fro')
dist_means_neg = la.norm(la.logm(dataset_1_neg_mean) - la.logm(dataset_3_neg_mean), 'fro')

foo = np.vstack((dataset_1_cov_data, dataset_3_cov_data))
foo = np.reshape(foo, (np.shape(foo)[0], 90, 90))

foo1 = np.array(map(lambda x: la.logm(x), foo))
foo1 = np.reshape(foo1, (250, 8100))

pca = PCA(n_components = 2)
foo2 = pca.fit_transform(foo1)

plt.scatter(foo2[:100, 0], foo2[:100, 1], color='blue')   
plt.scatter(foo2[100:, 0], foo2[100:, 1], color='red')