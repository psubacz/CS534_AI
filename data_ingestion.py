# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:28:30 2019

@author: Eastwind
"""
import csv
import numpy as np

def parse_data_file(data_file):
    '''
    Parses the data file as a tab seperated value file
    Returns an array of x-y values
    '''
    #Create nested lists for each attribute
    data = []
    X = []
    Y = []
    #Open the data file and generate a list of data from teh tsv
    with open(data_file) as tsv:
        for line in csv.reader(tsv, delimiter=","):
            #Encode the data to ensure numerical methods. Unknown data is kept
            line = encode_data(line)
            if line is not None:
                data.append(line)

    #The data is not time dependant and can be shuffled.
    for line in data:
        X.append(line[0:len(line)-1])
        Y.append([line[-1]])
    X = np.array(X)
    Y = np.array(Y)
    return X, np.reshape(np.array(Y),(Y.shape[0],))

def plug_missing_values(dataset):
#    '''
#    plugs missing data by calculating the mean feature for each feature set.
#    '''
    for i in range(0,dataset.shape[0]):
        counter = 0
        mean = 0
        for ii in range(0,dataset.shape[1]):
            if dataset[i][ii] is not None:
#                print(dataset[i][ii])
                counter +=1
                mean+=dataset[i][ii]
        mean/=counter
        
        for ii in range(0,dataset.shape[1]):
            if dataset[i][ii] is None:           
                dataset[i][ii] = mean
    return dataset

def encode_data(data):
    '''
    Encode data to be all numerical. Boolean esque values are set to 0 or 1 
     depending on code. 
    '''
    for i in range(0,len(data)):
        if (data[i]== '?'):
            data[i] = None
        else:
            data[i] = float(data[i])
    return data