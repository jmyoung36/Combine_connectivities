# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:16:19 2016

@author: jonyoung
"""

import numpy as np
from sklearn import svm, cross_validation, metrics
import connectivity_utils as utils
import pandas as pd
import csv
import scipy.linalg as la
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

# k-fold CV with a precomputed kernel, returning sensitivity and specificity as well as accuracy
def kcv(data, labels, classifier, n_folds):
    
    preds = np.zeros_like(labels)
    dvs = np.zeros_like(labels) 
    splits = cross_validation.KFold(len(labels), n_folds)
    
    for train_index, test_index in splits :
                
        K_train = data[train_index,:][:, train_index]
        K_test = data[test_index, :][:, train_index]
        labels_train = labels[train_index]
        classifier.fit(K_train, labels_train)
        preds[test_index] = classifier.predict(K_test)
        dvs[test_index] = classifier.decision_function(K_test)
        
    acc = metrics.accuracy_score(labels, preds)
    sens = float(sum(preds[labels == 1] == 1))/sum(labels == 1)
    spec = float(sum(preds[labels == 0] == 0))/sum(labels == 0)
                
    return acc, sens, spec
    
# set directories
dataset_1_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/'
dataset_3_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL3_timecourse/'

# indices of lower triangular elements
lotril_inds = [np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))]

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

# extract the lower triangle
dataset_1_edge_data_s = np.squeeze(dataset_1_cov_data_s[:, lotril_inds], axis=1)

# initialise the classifier
clf = svm.SVC(kernel='precomputed')

# make kernel and run 10-fold cross validated classification
K = np.dot(dataset_1_edge_data_s, np.transpose(dataset_1_edge_data_s))
K_1=K
acc, sens, spec = kcv(K, dataset_1_cov_labels_s, clf, 10)

print 'Results for dataset 1 in original space:'
print 'Accuracy = ' + str(acc)
print 'Sensitivity = ' + str(sens)
print 'Specificity = '+ str(spec)

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

# extract the lower triangle
dataset_3_edge_data_s = np.squeeze(dataset_3_cov_data_s[:, lotril_inds], axis=1)

# make kernel and run 10-fold cross validated classification
K = np.dot(dataset_3_edge_data_s, np.transpose(dataset_3_edge_data_s))
K_3=K
acc, sens, spec = kcv(K, dataset_3_cov_labels_s, clf, 10)

print 'Results for dataset 3 in original space:'
print 'Accuracy = ' + str(acc)
print 'Sensitivity = ' + str(sens)
print 'Specificity = '+ str(spec)

# combine the datasets and run a new classification
combined_datasets_edge_data = np.vstack((dataset_1_edge_data_s, dataset_3_edge_data_s))
combined_datasets_labels = np.hstack((dataset_1_cov_labels_s, dataset_3_cov_labels_s))
distinguish_datasets_labels = np.zeros((len(combined_datasets_labels), 1))
print np.shape(distinguish_datasets_labels)
distinguish_datasets_labels[0:len(dataset_1_cov_labels_s)]= 1
print distinguish_datasets_labels

# shuffle the rows and labels
r = np.random.permutation(len(combined_datasets_labels))
combined_datasets_edge_data_s = combined_datasets_edge_data[r, :]
combined_datasets_labels_s = combined_datasets_labels[r]
distinguish_datasets_labels_s = distinguish_datasets_labels[r]

# make kernel and run 10-fold cross validated classification
K = np.dot(combined_datasets_edge_data_s, np.transpose(combined_datasets_edge_data_s))

# do x-validated stacking prediction
kf = cross_validation.KFold(len(combined_datasets_labels_s), 10)

for train_index, test_index in kf :
    
    train_index_binary = np.zeros((len(combined_datasets_labels_s), 1))
    train_index_binary[train_index] = 1

    # train separate classifiers on the data from the different datasets
    train_index_ds_1 = np.bitwise_and(train_index_binary == 1, distinguish_datasets_labels_s == 1)
    train_index_ds_3 = np.bitwise_and(train_index_binary == 1, distinguish_datasets_labels_s == 0)

    
    # make the kernels and labels
    K_train_ds_1 = K[train_index_ds_1, :][:, train_index_ds_1]
    train_labels_ds_1 = combined_datasets_labels_s[train_index_ds_1]
    K_train_ds_3 = K[train_index_ds_3, :][:, train_index_ds_3]
    train_labels_ds_3 = combined_dataset_labels_s[train_ind]


