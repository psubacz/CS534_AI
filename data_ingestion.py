# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:28:30 2019

@author: Eastwind
"""
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing

def parse_data_file(data_file):
    # read in csv
    dataset = pd.read_csv(data_file)

    # extract actual classification from data
    y = dataset['Class']
    X = dataset.drop('Class', 1)

    # convert all values to numbers, NA if '?'
    for column in X:
        X[column] = pd.to_numeric(dataset[column], errors='coerce')
    
    # fill missing with mean
    X = X.fillna(X.mean())

    # normalize 
    X[:] = preprocessing.MinMaxScaler().fit_transform(X.values)

    return X, y


if __name__ == '__main__':
    parse_data_file('arrhythmia.csv')
