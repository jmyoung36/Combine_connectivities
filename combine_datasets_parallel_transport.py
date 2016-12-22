# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:53:11 2016

@author: jonyoung
"""

import numpy as np
from sklearn import svm, cross_validation, metrics, model_selection
import connectivity_utils as utils
import pandas as pd
import csv
import scipy.linalg as la
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

# references
# [1] Ng, Bernard, et al. "Transport on riemannian manifold for functional connectivity-based classification." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer International Publishing, 2014
# [2] Barachant, Alexandre, et al. "Riemannian geometry applied to BCI classification." International Conference on Latent Variable Analysis and Signal Separation. Springer Berlin Heidelberg, 2010.
# [3] Barachant, Alexandre, et al. "Classification of covariance matrices using a Riemannian-based kernel for BCI applications." Neurocomputing 112 (2013): 172-178.
# [4] Barachant, Alexandre, and Marco Congedo. "A Plug&Play P300 BCI Using Information Geometry." arXiv preprint arXiv:1409.0107 (2014).
# [5] Arsigny, Vincent, et al. "Fast and simple calculus on tensors in the log-Euclidean framework." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer Berlin Heidelberg, 2005.

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
    
    for i in range(2) :
        
        print base_cov_matrix[:5, :5]
    
        # project all matrices into the tangent space
        tangent_matrices = np.zeros_like(matrices)
        for j in range(len(matrices)) :
        
            tangent_matrices[j, :, :] = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.logm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), matrices[j, :, :],la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
    
        # calculate the tangent space mean
        tangent_space_base_cov_matrix = np.mean(tangent_matrices, axis=0)
        
        print tangent_space_base_cov_matrix[:5, :5]
        
        # project new tangent mean back to the manifold
        base_cov_matrix = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.expm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), tangent_space_base_cov_matrix ,la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
        
    # apply whitening transport and projection for training AND testing data
#    projected_matrices = np.zeros_like(matrices)   
#    base_cov_matrix_pow = la.fractional_matrix_power(base_cov_matrix, -0.5)
#    
#    for i in range(len(matrices)) :
#        projected_matrices[i, :, :] = la.logm(np.linalg.multi_dot([base_cov_matrix_pow, matrices[i, :, :], base_cov_matrix_pow]))
#       
#    return projected_matrices
    return base_cov_matrix

# Generate the tangent vector at B 'pointing toward' A, where A and B are both
# symmetric positive definite matrices in S++. This is the inverse operation of the Exp map.
# From [1]
def log_map(A, B) :
    
    B_pos = la.fractional_matrix_power(B, 0.5)
    B_neg = la.fractional_matrix_power(B, -0.5)
    #return np.linalg.multi_dot([B_pos, la.logm(np.linalg.multi_dot([B_neg, A, B_neg])), B_pos])
    log_prod = la.logm(B_neg.dot(A).dot(B_neg))
    return B_pos.dot(log_prod).dot(B_pos)
    
# Project the tangent vector at B 'pointing toward' A back into S++, where A 
# and B are both symmetric positive definite matrices. This is the inverse operation of the Log map.
# From [1]
def exp_map(A, B) :
    
    B_pos = la.fractional_matrix_power(B, 0.5)
    B_neg = la.fractional_matrix_power(B, -0.5)
    #return np.linalg.multi_dot([B_pos, la.expm(np.linalg.multi_dot([B_neg, A, B_neg])), B_pos])
    exp_prod = la.expm(B_neg.dot(A).dot(B_neg))
    return B_pos.dot(exp_prod).dot(B_pos)
    
# create a geodesic, i.e. local shortest path, from matrix B to matrix A where 
# A and B are both positive definite matrices on S++. Take a distance argument 
# d allowing us to travel e.g. half way from B to A, or twice the distance from
# B to A.
# From [1]
def make_geodesic(A, B, d):
    
    return exp_map(d * log_map(A, B), B)
    
# calculate the Log-Euclidean mean of a set of n dxd matrices in S++, by applying
# a matrix logarithm to project the matrices onto the tangent space at I,
# calculating the arithmetic mean, and then projecting it back into S++.
# Take a stack of n dxd matrices MM, as a numpy array with dimensions d x d x n
# From [1]
def log_Euclidean_mean(MM):
    
    # create a structure to store the tangent space matrices
    logm_MM = np.zeros_like(MM)
    
    # project all the matrices in MM into the tangent space
    for i in range(np.shape(MM)[2]) :
        
        logm_MM[:, :, i] = np.logm(MM[:, :, i])
        
    # take the mean and project back to S++
    mean_logm_MM = np.mean(logm_MM, axis=2)
    return la.expm(mean_logm_MM)
    
# calculate the geometric mean of a set of n dxd matrices in S++. Initially 
# calculate the Euclidean mean. Then project the matrices into the tangent
# space of the Euclidean mean with log_map, calculate the mean of the projected
# matrices, and project this back into S++ with exp_map to provide an updated
# estimate of the geometric mean. Repeat with this new estimate until the 
# estimate converges or the maximum number of iterations is reached.
# Take a stack of n dxd matrices MM, as a numpy array with dimensions n x d x d
# From [2, 3]
def geometric_mean(MM, tol=10e-10, max_iter=50):
    
    print 'Calculating geometric mean. Iteration:'
    
    # initialise variables
    # Euclidean mean as first estimate
    new_est = np.mean(MM, axis=0)
    # convergence criterion
    crit = np.finfo(np.float64).max
    k = 0
    # number of matrices
    n = np.shape(MM)[0]
    
    # start the loop
    while (crit > tol) and (k < max_iter):
        
        
        #print new_est[:5, :5]
        
        # update the current estimate
        current_est = new_est
        
        # project all the matrices into the tangent space of the current estimate
        tan_MM = np.zeros_like(MM)
        for i in range(n) :
            
            #tan_MM[i, :, :] = log_map(current_est, MM[i, :, :])
            tan_MM[i, :, :] = log_map(MM[i, :, :], current_est)
            
        # arithmetic mean in the tangent spacegeometric_mean
        S = np.mean(tan_MM, axis=0)
        
        #print S[:5, :5]
        
        # project S back to S++ to create a new estimated mean
        new_est = exp_map(S, current_est)
        #new_est = exp_map(current_est, S)
        
        # housekeeping: update k and crit
        k = k + 1
        #crit = np.linalg.norm(S, ord='fro')
        crit = np.linalg.norm(new_est - current_est, ord='fro')
        print k
        print crit
        
    return new_est

# take a source and a target dataset, each of connectivity matrices. Find their
# means (of a specified type) and then generate a discretised geodesic from
# the mean of the source dataset to the mean of the target dataset. Use this
# geodesic to parallel transport the connectivity matrices in the source 
# dataset to the location of the target dataset with Schild's Ladder.
def parallel_transport_dataset(source_dataset, target_dataset, mean_type='geometric', n_steps=10, target_mean=None) :
    
    # calculate the means
    if mean_type == 'geometric' :
        
        source_mean = geometric_mean(source_dataset)
        
        if target_mean == None :        
              
            target_mean = geometric_mean(target_dataset)
                
        else :
                
            target_mean = target_mean
        
    elif mean_type == 'log_Euclidean' :

        source_mean = log_Euclidean_mean(source_dataset)
        
        if target_mean == None :        
                
            target_mean = log_Euclidean_mean(target_dataset)
                
        else :
                
            target_mean = target_mean
        
    elif mean_type == 'Euclidean' :
        
        source_mean = np.mean(source_dataset, axis=0)
        
        if target_mean == None :        
                
            target_mean = np.mean(target_dataset, axis=0)
                
        else :
                
            target_mean = target_mean           
            
    else :
        
        print "Mean method not supported: Must be 'geometric' (default), 'log_Euclidean' or 'Euclidean'"
        return 0
        
    # generate a discretised geodesic with n_steps step, at source_mean and 
    # pointing toward target_mean
    disc_geo = np.zeros((n_steps, 90, 90))
    for i in range(n_steps) :
        
        frac_dist = (i+1)/float(n_steps)
        disc_geo[i, :, :] = make_geodesic(target_mean, source_mean, frac_dist)
        
    # initialise the transported_source_matrices
    transported_source_dataset = source_dataset
        
    # perform the transport
    n_source_matrices = np.shape(source_dataset)[0]
    # first n_steps - 1 steps
    for i in range(n_steps - 1) :
        
        # loop through all the matrices in the transported_source_dataset.
        for j in range(n_source_matrices) :
            
            # find the midpoint of the geodesic joining the jth transported
            # source matrix and the i+1th (next) point of disc_geo
            midpoint = make_geodesic(disc_geo[i+1, :, :], transported_source_dataset[j, :, :], 0.5)
            
            # find the new transported source matrix by moving twice the
            # distance from the ith point of disc geo to the midpoint
            transported_source_dataset[j, :, :] = make_geodesic(midpoint, disc_geo[i, :, :] , 2.0)
            
        print transported_source_dataset[0, :5, :5]
      
    # final step to the target dataset
    for j in range(n_source_matrices) :
            
        # find the midpoint of the geodesic joining the jth transported
        # source matrix and the target mean
        midpoint = make_geodesic(target_mean, transported_source_dataset[j, :, :], 0.5)
            
        # find the new transported source matrix by moving twice the
        # distance from the ith point of disc geo to the midpoint
        transported_source_dataset[j, :, :] = make_geodesic(midpoint, disc_geo[n_steps - 1, :, :] , 2.0)     
        
    print transported_source_dataset[0, :5, :5]
        
    # final output: transported source dataset after n_steps steps, plus source
    # and target means as they may be needed
    return transported_source_dataset, source_mean, target_mean
    
# geodesic distance between two matrices A and B on S++, according to
# information geometry
# From [4]
def IG_distance(A, B) :

    A_neg = la.fractional_matrix_power(A, -0.5)
    log_prod = la.logm(A_neg.dot(B).dot(A_neg))
    return np.linalg.norm(log_prod, ord='fro')
    
# geodesic distance between two matrices A and B on S++, according to log-
# Euclidean  metric
# From [5]
def log_Euclidean_distance(A, B) :

    diff_log = la.logm(A) - la.logm(B)
    return np.linalg.norm(diff_log, ord='fro')
        
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

# calculate the (geometric) means
# reshape
dataset_1_cov_data_s = np.reshape(dataset_1_cov_data_s, (100, 90, 90))
dataset_3_cov_data_s = np.reshape(dataset_3_cov_data_s, (150, 90, 90))
# get means
#dataset_1_geo_mean = geometric_mean(dataset_1_cov_data_s)
#dataset_3_geo_mean = geometric_mean(dataset_3_cov_data_s)

#parallel_transported_dataset, source_mean, target_mean = parallel_transport_dataset(dataset_1_cov_data_s, dataset_3_cov_data_s, mean_type='geometric', n_steps=500)
#parallel_transported_dataset_mean = geometric_mean(parallel_transported_dataset)
#parallel_transported_dataset_mean = np.mean(parallel_transported_dataset, axis=0)

parallel_transported_dataset_1, source_mean_1, target_mean = parallel_transport_dataset(dataset_1_cov_data_s, dataset_3_cov_data_s, mean_type='geometric', n_steps=20, target_mean=np.eye(90))
#parallel_transported_dataset_1_mean = geometric_mean(parallel_transported_dataset_1)
parallel_transported_dataset_3, source_mean_3, target_mean = parallel_transport_dataset(dataset_3_cov_data_s, dataset_3_cov_data_s, mean_type='geometric', n_steps=20, target_mean=np.eye(90))
#parallel_transported_dataset_3_mean = geometric_mean(parallel_transported_dataset_3)

# save the original and parallel transported datasets
#foo = np.reshape(dataset_1_cov_data_s, (100, 8100))
#np.savetxt('/home/jonyoung/IoP_data/Data/connectivity_data/dataset_1_cov_data.csv', np.hstack((np.expand_dims(dataset_1_cov_labels_s, axis=1), foo)), delimiter=',')
#foo = np.reshape(dataset_3_cov_data_s, (150, 8100))
#np.savetxt('/home/jonyoung/IoP_data/Data/connectivity_data/dataset_3_cov_data.csv', np.hstack((np.expand_dims(dataset_3_cov_labels_s, axis=1), foo)), delimiter=',')
#foo = np.reshape(parallel_transported_dataset_1, (100, 8100))
#np.savetxt('/home/jonyoung/IoP_data/Data/connectivity_data/dataset_1_parallel_transported_dataset_3.csv', np.hstack((np.expand_dims(dataset_1_cov_labels_s, axis=1), foo)), delimiter=',')
#foo = np.reshape(dataset_3_cov_data_s, (150, 8100))
#np.savetxt('/home/jonyoung/IoP_data/Data/connectivity_data/dataset_3_parallel_transported_I.csv', np.hstack((np.expand_dims(dataset_3_cov_labels_s, axis=1), foo)), delimiter=',')

# make a set of labels for the combined datasets
combined_datasets_labels_s = np.hstack((dataset_1_cov_labels_s, dataset_3_cov_labels_s))
distinguish_datasets_labels = np.zeros_like(combined_datasets_labels_s)
distinguish_datasets_labels[0:len(dataset_1_cov_labels_s)] = 1
dataset_1_pos = np.bitwise_and(distinguish_datasets_labels == 1, combined_datasets_labels_s == 1)
dataset_1_neg = np.bitwise_and(distinguish_datasets_labels == 1, combined_datasets_labels_s == 0)
dataset_3_pos = np.bitwise_and(distinguish_datasets_labels == 0, combined_datasets_labels_s == 1)
dataset_3_neg = np.bitwise_and(distinguish_datasets_labels == 0, combined_datasets_labels_s == 0)

# try Euclidean mean shift
#dataset_1_Euclidean_mean = np.mean(dataset_1_cov_data_s, axis=0)
#dataset_3_Euclidean_mean = np.mean(dataset_3_cov_data_s, axis=0)
#mean_shift_vector = dataset_3_Euclidean_mean - dataset_1_Euclidean_mean
#Euclidean_transported_dataset = np.zeros_like(dataset_1_cov_data_s)
#for i in range(np.shape(dataset_1_cov_data_s)[0]) :
#    Euclidean_transported_dataset[i, :, :] = dataset_1_cov_data_s[i, :, :] + mean_shift_vector
#
## combine the datasets and extract the lower triangle
#Euclidean_transported_dataset = np.reshape(Euclidean_transported_dataset, (100, 8100))
#dataset_3_cov_data_s = np.reshape(dataset_3_cov_data_s, (150, 8100))
#Euclidean_transported_combined_datasets = np.vstack((Euclidean_transported_dataset, dataset_3_cov_data_s))
#Euclidean_transported_combined_datasets = np.squeeze(Euclidean_transported_combined_datasets[:, lotril_inds])
#    
## do joint PCA on the combined data
#pca = PCA(n_components = 2)
#Euclidean_transported_combined_datasets_PCA = pca.fit_transform(Euclidean_transported_combined_datasets)
#
## plot the PCA with separate markers/colours for the datasets and labels
#plt.figure(0)
#plt.scatter(Euclidean_transported_combined_datasets_PCA[dataset_1_pos,0], Euclidean_transported_combined_datasets_PCA[dataset_1_pos,1], marker='+', color='blue')
#plt.scatter(Euclidean_transported_combined_datasets_PCA[dataset_1_neg,0], Euclidean_transported_combined_datasets_PCA[dataset_1_neg,1], marker='+', color='red')
#plt.scatter(Euclidean_transported_combined_datasets_PCA[dataset_3_pos,0], Euclidean_transported_combined_datasets_PCA[dataset_3_pos,1], marker='o', color='blue')
#plt.scatter(Euclidean_transported_combined_datasets_PCA[dataset_3_neg,0], Euclidean_transported_combined_datasets_PCA[dataset_3_neg,1], marker='o', color='red')
#
## try parallel transported data
## combine the datasets and extract the lower triangle
#parallel_transported_dataset = np.reshape(parallel_transported_dataset, (100, 8100))
##dataset_3_cov_data_s = np.reshape(dataset_3_cov_data_s, (150, 8100))
#parallel_transported_combined_datasets = np.vstack((parallel_transported_dataset, dataset_3_cov_data_s))
#parallel_transported_combined_datasets = np.squeeze(parallel_transported_combined_datasets[:, lotril_inds])
#    
## do joint PCA on the combined data
#pca = PCA(n_components = 2)
#parallel_transported_combined_datasets_PCA = pca.fit_transform(parallel_transported_combined_datasets)
#
## plot the PCA with separate markers/colours for the datasets and labels
#plt.figure(1)
#plt.scatter(parallel_transported_combined_datasets_PCA[dataset_1_pos,0], parallel_transported_combined_datasets_PCA[dataset_1_pos,1], marker='+', color='blue')
#plt.scatter(parallel_transported_combined_datasets_PCA[dataset_1_neg,0], parallel_transported_combined_datasets_PCA[dataset_1_neg,1], marker='+', color='red')
#plt.scatter(parallel_transported_combined_datasets_PCA[dataset_3_pos,0], parallel_transported_combined_datasets_PCA[dataset_3_pos,1], marker='o', color='blue')
#plt.scatter(parallel_transported_combined_datasets_PCA[dataset_3_neg,0], parallel_transported_combined_datasets_PCA[dataset_3_neg,1], marker='o', color='red')

# classification experiments - native space
# initialise the classifier
clf = svm.SVC(kernel='precomputed')

combined_datasets = np.vstack((dataset_1_cov_data_s, dataset_3_cov_data_s))
combined_datasets = np.squeeze(np.reshape(combined_datasets, (250, 8100))[:, lotril_inds])

K_dataset_1 = np.dot(combined_datasets[:100, :], np.transpose(combined_datasets[:100, :]))
K_dataset_3 = np.dot(combined_datasets[100:, :], np.transpose(combined_datasets[100:, :]))
K_combined = np.dot(combined_datasets, np.transpose(combined_datasets))

dataset_1_accs = model_selection.cross_val_score(clf, K_dataset_1, dataset_1_cov_labels_s, cv=10)
dataset_3_accs = model_selection.cross_val_score(clf, K_dataset_3, dataset_3_cov_labels_s, cv=10)
combined_datasets_accs = model_selection.cross_val_score(clf, K_combined, combined_datasets_labels_s, cv=10)
print 'Raw connectivities: dataset 1 acc = ' + str(np.mean(dataset_1_accs))
print 'Raw connectivities: dataset 3 acc = ' + str(np.mean(dataset_3_accs))
print 'Raw connectivities: joint acc = ' + str(np.mean(combined_datasets_accs))
#
## classification experiments - Log-Euclidean
#combined_datasets = np.vstack((dataset_1_cov_data_s, dataset_3_cov_data_s))
#for i in range(np.shape(combined_datasets)[0]) :
#    combined_datasets[i, :, :] = la.logm( combined_datasets[i, :, :])  
#combined_datasets = np.squeeze(np.reshape(combined_datasets, (250, 8100))[:, lotril_inds])
#
#K_dataset_1 = np.dot(combined_datasets[:100, :], np.transpose(combined_datasets[:100, :]))
#K_dataset_3 = np.dot(combined_datasets[100:, :], np.transpose(combined_datasets[100:, :]))
#K_combined = np.dot(combined_datasets, np.transpose(combined_datasets))
#
#dataset_1_accs = model_selection.cross_val_score(clf, K_dataset_1, dataset_1_cov_labels_s, cv=10)
#dataset_3_accs = model_selection.cross_val_score(clf, K_dataset_3, dataset_3_cov_labels_s, cv=10)
#combined_datasets_accs = model_selection.cross_val_score(clf, K_combined, combined_datasets_labels_s, cv=10)
#print 'Log-Euclidean connectivities: dataset 1 acc = ' + str(np.mean(dataset_1_accs))
#print 'Log-Euclidean connectivities: dataset 3 acc = ' + str(np.mean(dataset_3_accs))
#print 'Log-Euclidean connectivities: joint acc = ' + str(np.mean(combined_datasets_accs))
#
## classification experiments - tangent space of dataset 3 mean
#dataset_3_geo_mean = geometric_mean(dataset_3_cov_data_s)
#combined_datasets = np.vstack((dataset_1_cov_data_s, dataset_3_cov_data_s))
#for i in range(np.shape(combined_datasets)[0]) :
#    combined_datasets[i, :, :] = log_map(combined_datasets[i, :, :], dataset_3_geo_mean)
#combined_datasets = np.squeeze(np.reshape(combined_datasets, (250, 8100))[:, lotril_inds])
#
#K_dataset_1 = np.dot(combined_datasets[:100, :], np.transpose(combined_datasets[:100, :]))
#K_dataset_3 = np.dot(combined_datasets[100:, :], np.transpose(combined_datasets[100:, :]))
#K_combined = np.dot(combined_datasets, np.transpose(combined_datasets))
#
#dataset_1_accs = model_selection.cross_val_score(clf, K_dataset_1, dataset_1_cov_labels_s, cv=10)
#dataset_3_accs = model_selection.cross_val_score(clf, K_dataset_3, dataset_3_cov_labels_s, cv=10)
#combined_datasets_accs = model_selection.cross_val_score(clf, K_combined, combined_datasets_labels_s, cv=10)
#print 'dataset 3 geometric mean tangent connectivities: dataset 1 acc = ' + str(np.mean(dataset_1_accs))
#print 'dataset 3 geometric mean tangent connectivities: dataset 3 acc = ' + str(np.mean(dataset_3_accs))
#print 'dataset 3 geometric mean tangent connectivities: joint acc = ' + str(np.mean(combined_datasets_accs))
#
# classification experiments - Euclidean mean shift
dataset_1_mean = np.mean(dataset_1_cov_data_s, axis=0)
dataset_3_mean = np.mean(dataset_3_cov_data_s, axis=0)
mean_shift_vector_1 = np.eye(90) - dataset_1_mean
mean_shift_vector_3 = np.eye(90) - dataset_3_mean
combined_datasets = np.vstack((dataset_1_cov_data_s, dataset_3_cov_data_s))
for i in range(np.shape(dataset_1_cov_data_s)[0]) :
    i
    combined_datasets[i, :, :] = combined_datasets[i, :, :] + mean_shift_vector_1
for i in range(np.shape(dataset_1_cov_data_s)[0], np.shape(dataset_1_cov_data_s)[0] + np.shape(dataset_3_cov_data_s)[0]) :
    i
    combined_datasets[i, :, :] = combined_datasets[i, :, :] + mean_shift_vector_3
combined_datasets = np.squeeze(np.reshape(combined_datasets, (250, 8100))[:, lotril_inds])

K_dataset_1 = np.dot(combined_datasets[:100, :], np.transpose(combined_datasets[:100, :]))
K_dataset_3 = np.dot(combined_datasets[100:, :], np.transpose(combined_datasets[100:, :]))
K_combined = np.dot(combined_datasets, np.transpose(combined_datasets))

dataset_1_accs = model_selection.cross_val_score(clf, K_dataset_1, dataset_1_cov_labels_s, cv=10)
dataset_3_accs = model_selection.cross_val_score(clf, K_dataset_3, dataset_3_cov_labels_s, cv=10)
combined_datasets_accs = model_selection.cross_val_score(clf, K_combined, combined_datasets_labels_s, cv=10)
print 'Euclidean mean shift connectivities: dataset 1 acc = ' + str(np.mean(dataset_1_accs))
print 'Euclidean mean shift connectivities: dataset 3 acc = ' + str(np.mean(dataset_3_accs))
print 'Euclidean mean shift connectivities: joint acc = ' + str(np.mean(combined_datasets_accs))

# classification experiments - parallel transport
#print 'Distance from mean of source dataset to mean of target dataset (information geometry) = ' + str(IG_distance(source_mean, target_mean))
#print 'Distance from mean of source dataset to mean of target dataset (log-Euclidean) = ' + str(log_Euclidean_distance(source_mean, target_mean))
#print 'Distance from mean of transported source dataset to mean of target dataset (information geometry) = ' + str(IG_distance(parallel_transported_dataset_mean, target_mean))
#print 'Distance from mean of transported source dataset to mean of target dataset (log-Euclidean) = ' + str(log_Euclidean_distance(parallel_transported_dataset_mean, target_mean))
#
#combined_datasets = np.vstack((parallel_transported_dataset, dataset_3_cov_data_s))
#combined_datasets = np.reshape(combined_datasets, (250, 8100))
#combined_datasets = np.squeeze(np.reshape(combined_datasets, (250, 8100))[:, lotril_inds])
#
#K_dataset_1 = np.dot(combined_datasets[:100, :], np.transpose(combined_datasets[:100, :]))
#K_dataset_3 = np.dot(combined_datasets[100:, :], np.transpose(combined_datasets[100:, :]))
#K_combined = np.dot(combined_datasets, np.transpose(combined_datasets))
#
#dataset_1_accs = model_selection.cross_val_score(clf, K_dataset_1, dataset_1_cov_labels_s, cv=10)
#dataset_3_accs = model_selection.cross_val_score(clf, K_dataset_3, dataset_3_cov_labels_s, cv=10)
#combined_datasets_accs = model_selection.cross_val_score(clf, K_combined, combined_datasets_labels_s, cv=10)
#print 'Parallel transported connectivities: dataset 1 acc = ' + str(np.mean(dataset_1_accs))
#print 'Parallel transported connectivities: dataset 3 acc = ' + str(np.mean(dataset_3_accs))
#print 'Parallel transported connectivities: joint acc = ' + str(np.mean(combined_datasets_accs))
#
## classification experiments - parallel transport in tangent space of dataset 3
## geometric mean
#combined_datasets = np.vstack((parallel_transported_dataset, dataset_3_cov_data_s))
#for i in range(np.shape(combined_datasets)[0]) :
#    combined_datasets[i, :, :] = log_map(combined_datasets[i, :, :], target_mean)
#combined_datasets = np.reshape(combined_datasets, (250, 8100))
#combined_datasets = np.squeeze(np.reshape(combined_datasets, (250, 8100))[:, lotril_inds])
#
#K_dataset_1 = np.dot(combined_datasets[:100, :], np.transpose(combined_datasets[:100, :]))
#K_dataset_3 = np.dot(combined_datasets[100:, :], np.transpose(combined_datasets[100:, :]))
#K_combined = np.dot(combined_datasets, np.transpose(combined_datasets))
#
#dataset_1_accs = model_selection.cross_val_score(clf, K_dataset_1, dataset_1_cov_labels_s, cv=10)
#dataset_3_accs = model_selection.cross_val_score(clf, K_dataset_3, dataset_3_cov_labels_s, cv=10)
#combined_datasets_accs = model_selection.cross_val_score(clf, K_combined, combined_datasets_labels_s, cv=10)
#print 'dataset 3 geometric mean tangent parallel transported connectivities: dataset 1 acc = ' + str(np.mean(dataset_1_accs))
#print 'dataset 3 geometric mean tangent parallel transported connectivities: dataset 3 acc = ' + str(np.mean(dataset_3_accs))
#print 'dataset 3 geometric mean tangent parallel transported connectivities: joint acc = ' + str(np.mean(combined_datasets_accs))

#print 'Distance from mean of source dataset 1 to the identity (information geometry) = ' + str(IG_distance(source_mean_1, np.eye(90)))
#print 'Distance from mean of source dataset 1 to the identity (log-Euclidean) = ' + str(log_Euclidean_distance(source_mean_1, np.eye(90)))
#print 'Distance from mean of transported source dataset 1 to the identity (information geometry) = ' + str(IG_distance(parallel_transported_dataset_1_mean, np.eye(90)))
#print 'Distance from mean of transported source dataset 1 to the identity (log-Euclidean) = ' + str(log_Euclidean_distance(parallel_transported_dataset_1_mean, np.eye(90)))
#print 'Distance from mean of source dataset 3 to the identity (information geometry) = ' + str(IG_distance(source_mean_3, np.eye(90)))
#print 'Distance from mean of source dataset 3 to the identity (log-Euclidean) = ' + str(log_Euclidean_distance(source_mean_3, np.eye(90)))
#print 'Distance from mean of transported source dataset 3 to the identity (information geometry) = ' + str(IG_distance(parallel_transported_dataset_3_mean, np.eye(90)))
#print 'Distance from mean of transported source dataset 3 to the identity (log-Euclidean) = ' + str(log_Euclidean_distance(parallel_transported_dataset_3_mean, np.eye(90)))


combined_datasets = np.vstack((parallel_transported_dataset_1, parallel_transported_dataset_3))
combined_datasets_tangent = np.zeros_like(combined_datasets)
for i in range(np.shape(combined_datasets)[0]) :
    combined_datasets_tangent[i, :, :] = la.logm(combined_datasets[i, :, :])
combined_datasets = np.reshape(combined_datasets, (250, 8100))
combined_datasets_lotril = np.squeeze(np.reshape(combined_datasets, (250, 8100))[:, lotril_inds])
combined_datasets_tangent = np.reshape(combined_datasets_tangent, (250, 8100))
combined_datasets_tangent_lotril = np.squeeze(np.reshape(combined_datasets_tangent, (250, 8100))[:, lotril_inds])

K_dataset_1 = np.dot(combined_datasets_lotril[:100, :], np.transpose(combined_datasets_lotril[:100, :]))
K_dataset_3 = np.dot(combined_datasets_lotril[100:, :], np.transpose(combined_datasets_lotril[100:, :]))
K_combined = np.dot(combined_datasets_lotril, np.transpose(combined_datasets_lotril))

dataset_1_accs = model_selection.cross_val_score(clf, K_dataset_1, dataset_1_cov_labels_s, cv=10)
dataset_3_accs = model_selection.cross_val_score(clf, K_dataset_3, dataset_3_cov_labels_s, cv=10)
combined_datasets_accs = model_selection.cross_val_score(clf, K_combined, combined_datasets_labels_s, cv=10)
print 'parallel transported to I connectivities: dataset 1 acc = ' + str(np.mean(dataset_1_accs))
print 'parallel transported to I connectivities: dataset 3 acc = ' + str(np.mean(dataset_3_accs))
print 'parallel transported to I connectivities: joint acc = ' + str(np.mean(combined_datasets_accs))

K_dataset_1 = np.dot(combined_datasets_tangent_lotril[:100, :], np.transpose(combined_datasets_tangent_lotril[:100, :]))
K_dataset_3 = np.dot(combined_datasets_tangent_lotril[100:, :], np.transpose(combined_datasets_tangent_lotril[100:, :]))
K_combined = np.dot(combined_datasets_tangent_lotril, np.transpose(combined_datasets_tangent_lotril))

dataset_1_accs = model_selection.cross_val_score(clf, K_dataset_1, dataset_1_cov_labels_s, cv=10)
dataset_3_accs = model_selection.cross_val_score(clf, K_dataset_3, dataset_3_cov_labels_s, cv=10)
combined_datasets_accs = model_selection.cross_val_score(clf, K_combined, combined_datasets_labels_s, cv=10)
print 'parallel transported to I connectivities in tangent space of I: dataset 1 acc = ' + str(np.mean(dataset_1_accs))
print 'parallel transported to I connectivities in tangent space of I: dataset 3 acc = ' + str(np.mean(dataset_3_accs))
print 'parallel transported to I connectivities in tangent space of I: joint acc = ' + str(np.mean(combined_datasets_accs))