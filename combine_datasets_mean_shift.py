# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:12:34 2016

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

# visualise differences between datasets
datasets_combined = np.vstack((dataset_1_cov_data_s, dataset_3_cov_data_s))
connectivities_combined = np.reshape(datasets_combined, (np.shape(datasets_combined)[0], 90, 90))
log_connectivities_combined = np.array(map(lambda x: la.logm(x), connectivities_combined))
log_datasets_combined = np.reshape(log_connectivities_combined, (250, 8100))

pca = PCA(n_components = 2)
X = pca.fit_transform(log_datasets_combined)

plt.scatter(X[:100, 0], X[:100, 1], color='blue')   
plt.scatter(X[100:, 0], X[100:, 1], color='red')

# join the labels
combined_datasets_labels_s = np.hstack((dataset_1_cov_labels_s, dataset_3_cov_labels_s))

# seperate and joint accuracy
clf = svm.SVC(kernel='linear')
scores = cross_validation.cross_val_score(clf, dataset_1_cov_data_s, dataset_1_cov_labels_s, cv=10)
print 'No correction:'
print 'dataset 1 accuracy = ' + str(np.mean(scores))
scores = cross_validation.cross_val_score(clf, dataset_3_cov_data_s, dataset_3_cov_labels_s, cv=10)
print 'dataset 3 accuracy = ' + str(np.mean(scores))
scores = cross_validation.cross_val_score(clf, datasets_combined, combined_datasets_labels_s, cv=10)
print 'Combined datasets accuracy = ' + str(np.mean(scores))

# do simple mean-shift of dataset 1 to dataset 3
dataset_1_mean = np.mean(dataset_1_cov_data, axis=0)
dataset_3_mean = np.mean(dataset_3_cov_data, axis=0)

# shift from mean of dataset 1 to mean of dataset 3
mean_shift = dataset_3_mean - dataset_1_mean

# shift dataset 1
dataset_1_cov_data_ms = dataset_1_cov_data_s + np.tile(mean_shift, (np.shape(dataset_1_cov_data_s)[0], 1))

# combine shifted dataset 1 with original dataset 3
shifted_datasets_combined = np.vstack((dataset_1_cov_data_ms, dataset_3_cov_data_s))

print 'Mean-shift correction:'
scores = cross_validation.cross_val_score(clf, shifted_datasets_combined , combined_datasets_labels_s, cv=10)
print 'Combined datasets accuracy = ' + str(np.mean(scores))

shifted_connectivities_combined = np.reshape(shifted_datasets_combined, (np.shape(shifted_datasets_combined)[0], 90, 90))
shifted_log_connectivities_combined = np.array(map(lambda x: la.logm(x), shifted_connectivities_combined))
shifted_log_datasets_combined = np.reshape(shifted_log_connectivities_combined, (250, 8100))

pca = PCA(n_components = 2)
X = pca.fit_transform(shifted_log_datasets_combined)

plt.figure()
plt.scatter(X[:100, 0], X[:100, 1], color='blue')   
plt.scatter(X[100:, 0], X[100:, 1], color='red')