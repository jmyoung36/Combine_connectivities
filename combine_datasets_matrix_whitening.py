# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:54:36 2016

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
    splits = cross_validation.KFold(len(labels), n_folds)
    
    for train_index, test_index in splits :
                
        K_train = data[train_index,:][:, train_index]
        K_test = data[test_index, :][:, train_index]
        labels_train = labels[train_index]
        classifier.fit(K_train, labels_train)
        preds[test_index] = classifier.predict(K_test)
        
    acc = metrics.accuracy_score(labels, preds)
    sens = float(sum(preds[labels == 1] == 1))/sum(labels == 1)
    spec = float(sum(preds[labels == 0] == 0))/sum(labels == 0)
                
    return acc, sens, spec
    
# calculate the geometric mean of a set of covariance matrices and use this to project
# the matrices into the tangent space of the mean matrix
def project_to_mean_tangent(matrices) :
    
    # find geometric mean
    # construct base covariance matrix by repeated averaging in tangent space
    # first, initialise base covariance matrix
    base_cov_matrix = np.mean(matrices, axis=0)
    print base_cov_matrix
    
    for i in range(20) :
    
        # project all matrices into the tangent space
        tangent_matrices = np.zeros_like(matrices)
        for j in range(len(matrices)) :
        
            tangent_matrices[j, :, :] = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.logm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), matrices[j, :, :],la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
    
        # calculate the tangent space mean
        tangent_space_base_cov_matrix = np.mean(tangent_matrices, axis=0)
        
        # project new tangent mean back to the manifold
        base_cov_matrix = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.expm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), tangent_space_base_cov_matrix ,la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
        
    # apply whitening transport and projection for training AND testing data
    projected_matrices = np.zeros_like(matrices)   
    base_cov_matrix_pow = la.fractional_matrix_power(base_cov_matrix, -0.5)
    
    for i in range(len(matrices)) :
        projected_matrices[i, :, :] = la.logm(np.linalg.multi_dot([base_cov_matrix_pow, matrices[i, :, :], base_cov_matrix_pow]))
       
    return projected_matrices
        
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

# shuffle the rows and labels
r = np.random.permutation(len(combined_datasets_labels))
combined_datasets_edge_data_s = combined_datasets_edge_data[r, :]
combined_datasets_labels_s = combined_datasets_labels[r]

# make kernel and run 10-fold cross validated classification
K = np.dot(combined_datasets_edge_data_s, np.transpose(combined_datasets_edge_data_s))
acc, sens, spec = kcv(K, combined_datasets_labels_s, clf, 10)

print 'Results for pooled dataset in original space:'
print 'Accuracy = ' + str(acc)
print 'Sensitivity = ' + str(sens)
print 'Specificity = '+ str(spec)

# how well can we tell apart the two datasets?
distinguish_datasets_labels = np.zeros_like(combined_datasets_labels)
distinguish_datasets_labels[0:len(dataset_1_cov_labels)] = 1
distinguish_datasets_labels_s = distinguish_datasets_labels[r]

acc, sens, spec = kcv(K, distinguish_datasets_labels_s, clf, 10)

print 'Results for distinguishing the datasets in the original space:'
print 'Accuracy = ' + str(acc)
print 'Sensitivity = ' + str(sens)
print 'Specificity = '+ str(spec)

# do matrix whitening transport separately on the two datasets so they are projected into
# the tangent space of their own mean matrix

# reshape both datasets into sets of matrices
#dataset_1_cov_matrices = np.reshape(dataset_1_cov_data, (len(dataset_1_cov_data), 90, 90))
#dataset_3_cov_matrices = np.reshape(dataset_3_cov_data, (len(dataset_3_cov_data), 90, 90))
#
## do projection
#projected_dataset_1_cov_matrices = project_to_mean_tangent(dataset_1_cov_matrices)
#projected_dataset_3_cov_matrices = project_to_mean_tangent(dataset_3_cov_matrices)


# project both sets of matrices into tangent space of identity matrix
# do projection
#projected_dataset_1_cov_data = np.apply_along_axis(lambda x: np.reshape(la.logm(np.reshape(x, (90, 90))), (8100)), 1, dataset_1_cov_data)
#projected_dataset_3_cov_data = np.apply_along_axis(lambda x: np.reshape(la.logm(np.reshape(x, (90, 90))), (8100)), 1, dataset_3_cov_data)
#
## extract the lower triangle from the projected matrices
#projected_dataset_1_edge_data = np.squeeze(projected_dataset_1_cov_data[:, lotril_inds], axis=1)
#projected_dataset_3_edge_data = np.squeeze(projected_dataset_3_cov_data[:, lotril_inds], axis=1)

# shuffle the rows and labels
#r = np.random.permutation(len(dataset_1_cov_labels))
#projected_dataset_1_edge_data_s = projected_dataset_1_edge_data[r, :]
#dataset_1_cov_labels_s = dataset_1_cov_labels[r]
#
## make kernel and run 10-fold cross validated classification
#K = np.dot(projected_dataset_1_edge_data_s, np.transpose(projected_dataset_1_edge_data_s))
#acc, sens, spec = kcv(K, dataset_1_cov_labels_s, clf, 10)
#
#print 'Results for dataset 1 in projected space:'
#print 'Accuracy = ' + str(acc)
#print 'Sensitivity = ' + str(sens)
#print 'Specificity = '+ str(spec)
#
## shuffle the rows and labels
#r = np.random.permutation(len(dataset_3_cov_labels))
#projected_dataset_3_edge_data_s = projected_dataset_3_edge_data[r, :]
#dataset_3_cov_labels_s = dataset_3_cov_labels[r]
#
## make kernel and run 10-fold cross validated classification
#K = np.dot(projected_dataset_3_edge_data_s, np.transpose(projected_dataset_3_edge_data_s))
#acc, sens, spec = kcv(K, dataset_3_cov_labels_s, clf, 10)
#
#print 'Results for dataset 3 in projected space:'
#print 'Accuracy = ' + str(acc)
#print 'Sensitivity = ' + str(sens)
#print 'Specificity = '+ str(spec)
#
## combine the projected_datasets and run a new classification
#projected_combined_datasets_edge_data = np.vstack((projected_dataset_1_edge_data_s, projected_dataset_3_edge_data_s))
#combined_datasets_labels = np.hstack((dataset_1_cov_labels_s, dataset_3_cov_labels_s))
#
## shuffle the rows and labels
#r = np.random.permutation(len(combined_datasets_labels))
#projected_combined_datasets_edge_data_s = projected_combined_datasets_edge_data[r, :]
#combined_datasets_labels_s = combined_datasets_labels[r]
#
## make kernel and run 10-fold cross validated classification
#K = np.dot(projected_combined_datasets_edge_data_s, np.transpose(projected_combined_datasets_edge_data_s))
#acc, sens, spec = kcv(K, combined_datasets_labels_s, clf, 10)
#
#print 'Results for pooled dataset in projected space:'
#print 'Accuracy = ' + str(acc)
#print 'Sensitivity = ' + str(sens)dataset_1_cov_matrices = np.reshape(dataset_1_cov_data, (len(dataset_1_cov_data), 90, 90))
#dataset_3_cov_matrices = np.reshape(dataset_3_cov_data, (len(dataset_3_cov_data), 90, 90))

# do projection
#projected_dataset_1_cov_matrices = project_to_mean_tangent(dataset_1_cov_matrices)
#projected_dataset_3_cov_matrices = project_to_mean_tangent(dataset_3_cov_matrices)
#print 'Specificity = '+ str(spec)

# try ordinary PCA
#kpca = KernelPCA(kernel='precomputed', n_components = 50)
#dataset_1_pca = kpca.fit_transform(K_1)
#dataset_3_pca = kpca.fit_transform(K_3)
#
## combined the datasets and run a new classification
#combined_datasets_pca = np.vstack((dataset_1_pca, dataset_3_pca))
#combined_datasets_labels = np.hstack((dataset_1_cov_labels_s, dataset_3_cov_labels_s))
#
## shuffle the rows and labels
#r = np.random.permutation(len(combined_datasets_labels))
#combined_datasets_pca_s = combined_datasets_pca[r, :]
#combined_datasets_labels_s = combined_datasets_labels[r]
#
## make kernel and run 10-fold cross validated classification
#K = np.dot(combined_datasets_pca_s, np.transpose(combined_datasets_pca_s))
#acc, sens, spec = kcv(K, combined_datasets_labels_s, clf, 10)
#print 'Results for pooled dataset in pca space:'
#print 'Accuracy = ' + str(acc)
#print 'Sensitivity = ' + str(sens)
#print 'Specificity = '+ str(spec)

# joint PCA
pca = PCA(n_components = 2)
combined_data_pca = pca.fit_transform(combined_datasets_edge_data_s)
dataset_1_pos = np.bitwise_and(distinguish_datasets_labels_s == 1, combined_datasets_labels_s == 1)
dataset_1_neg = np.bitwise_and(distinguish_datasets_labels_s == 1, combined_datasets_labels_s == 0)
dataset_3_pos = np.bitwise_and(distinguish_datasets_labels_s == 0, combined_datasets_labels_s == 1)
dataset_3_neg = np.bitwise_and(distinguish_datasets_labels_s == 0, combined_datasets_labels_s == 0)
plt.scatter(combined_data_pca[dataset_1_pos,0], combined_data_pca[dataset_1_pos,1], marker='+', color='blue')
plt.scatter(combined_data_pca[dataset_1_neg,0], combined_data_pca[dataset_1_neg,1], marker='+', color='red')
plt.scatter(combined_data_pca[dataset_3_pos,0], combined_data_pca[dataset_3_pos,1], marker='o', color='blue')
plt.scatter(combined_data_pca[dataset_3_neg,0], combined_data_pca[dataset_3_neg,1], marker='o', color='red')

print dataset_1_pos