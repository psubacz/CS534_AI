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

    # remove classifications without enough data ( < 4 instances)
    dataset.drop(dataset[dataset.Class == 7].index, inplace=True)
    dataset.drop(dataset[dataset.Class == 8].index, inplace=True)
    dataset.drop(dataset[dataset.Class == 14].index, inplace=True)
    dataset.drop(dataset[dataset.Class == 15].index, inplace=True)

    # shift classification indicies to start from 0
    dataset.loc[dataset.Class == 16, 'Class'] = 0
    dataset.loc[dataset.Class == 9, 'Class'] = 7
    dataset.loc[dataset.Class == 10, 'Class'] = 8

    # extract actual classification from data
    y = dataset['Class']
    X = dataset.drop('Class', 1)

    count = y.value_counts(dropna=False)

    # convert all values to numbers, NA if '?'
    for column in X:
        X[column] = pd.to_numeric(dataset[column], errors='coerce')

    # fill missing with mean
    X = X.fillna(X.mean())

    # normalize 
    X[:] = preprocessing.MinMaxScaler().fit_transform(X.values)

    return X, y, max(dataset['Class']) + 1


if __name__ == '__main__':
    parse_data_file('arrhythmia.csv')
