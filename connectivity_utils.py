# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:18:19 2016

@author: jonyoung
"""

# utilities used by all kernel generation algorithms

# import the packages we need
import numpy as np
import glob
import re
from sklearn.preprocessing import scale

# define natural sort key so we sort files into correct (natural) order
# taken from http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort?lq=1
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

# take a directory containing .txt files containing connectivity data and a 
# .csv file with the per-subject labels. Load the connectvity data into a numpy
# matrix with one row per subject and return this along with the set of labels
def load_connectivity_data(data_dir, standardise=True):
    
    # list of data files
    connectivity_files = glob.glob(data_dir + '*.txt')
    n_files = len(connectivity_files)
    
    # sort into numerical order
    connectivity_files.sort(key=natural_sort_key)    
    
    # generate numpy arrays for data & labels
    # each subject is 90 x 90 matrix = 8100 elements
    connectivity_data = np.zeros((n_files, 8100))
    
    # roll through files in order
    for connectivity_file, i in zip(connectivity_files, range(len(connectivity_files))) :
        
        with open(connectivity_file) as infile:
            connectivity_file_data = infile.readlines()
        connectivity_file_data = map(lambda x: x.split('\t'), connectivity_file_data)

        
        # tidy up - split into indidual elements, remove returns/newlines, convert 
        # to float, convert to np array, set diagonals to zero, and reshape
        connectivity_file_data = np.array(map(lambda x: [float(element) for element in x if not element == '\r\n'], connectivity_file_data))
        #connectivity_file_data[np.diag_indices(90)] = 0
        connectivity_data[i, :] = np.reshape(connectivity_file_data, (1, 8100))
        
        # optional standardisation of variables with z-transform
        #if standardise :
            
         #   connectivity_data = scale(connectivity_data, axis=0) 
        
    return connectivity_data, connectivity_files
    
def load_hcp_matrix(filename): 
    
    # open file and read in line by line
    with open(filename) as infile:
         
         connectivity_file_data = infile.readlines()
         
    # split on whitespace
    connectivity_file_data = map(lambda x: x.split(), connectivity_file_data)
    
    # convert to float
    connectivity_file_data = np.array(map(lambda x: [float(element) for element in x], connectivity_file_data))
    
    
    return connectivity_file_data
    
def load_hcp_timecourse(filename): 
    
    # open file and read in line by line
    with open(filename) as infile:
         
         connectivity_file_data = infile.readlines()
         
    # split on whitespace
    connectivity_file_data = map(lambda x: x.split(), connectivity_file_data)
    
    # convert to float
    connectivity_file_data = np.array(map(lambda x: [float(element) for element in x], connectivity_file_data))
    
    
    return connectivity_file_data   
    
# take a directory containing .txt files containing timecourse data and a 
# .csv file with the per-subject labels. Load the connectvity data into a 3d
# numpy array of number of subjectsx number of regions x number of timepoints
def load_timecourse_data(data_dir):
    
    # list of data files
    timecourse_files = glob.glob(data_dir + '*.txt')
    n_files = len(timecourse_files)
    
    # sort into numerical order
    timecourse_files.sort(key=natural_sort_key)    
    
    # generate numpy arrays for data & labels
    # each subject is 90 regions x 140 timepoints
    timecourse_data = np.zeros((n_files, 90, 190))
    
    # empty list of files actually used
    final_files = []
    
    # roll through files in order
    for timecourse_file, i in zip(timecourse_files, range(len(timecourse_files))) :
        
        print timecourse_file
        
        timecourse_data[i, :, :] = np.transpose(np.genfromtxt(timecourse_file, delimiter='\t')[:, :90])
        
        final_files = final_files + [timecourse_file]
        
    return timecourse_data, final_files
    
def load_labels(data_dir) :
            
    # read in label data and convert to np array
    with open(data_dir + 'Labels.csv') as infile:
        labels = infile.readlines()
    labels = np.array(map(lambda label: int(label[0]), labels))
    return labels
        
#foo = load_timecourse_data('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/')

    
    
    