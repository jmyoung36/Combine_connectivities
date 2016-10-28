# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:12:34 2016

@author: jonyoung
"""

import numpy as np
from sklearn import svm, cross_validation, metrics, manifold
import connectivity_utils as utils
import pandas as pd
import scipy.linalg as la

# indices of lower triangular elements
lotril_inds = [np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))]

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

# create a linear kernel classifier and use it
clf = svm.SVC(kernel='linear')
clf.fit(dataset_1_cov_data_s, dataset_1_cov_labels_s)

linear_weights = clf.coef_
linear_support_vectors = clf.support_vectors_

# create a linear kernel matrix
K = np.dot(dataset_1_cov_data_s, np.transpose(dataset_1_cov_data_s))

# create a precomputed kernel classifier and use it
clf = svm.SVC(kernel='precomputed')
clf.fit(K, dataset_1_cov_labels_s)
precomputed_support = clf.support_
precomputed_dual_coef = np.squeeze(clf.dual_coef_)
precomputed_support_vectors = dataset_1_cov_data_s[precomputed_support, :]
precomputed_weights = np.sum(np.apply_along_axis(lambda x: x * np.transpose(precomputed_dual_coef), 0, precomputed_support_vectors), axis=0)