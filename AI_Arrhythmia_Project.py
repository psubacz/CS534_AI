# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:30:30 2019

@author: Eastwind
"""
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import data_ingestion as di
import copy
#Scikit-Learn Modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


for i in range(10):

    pipe = Pipeline([('classifier' , RandomForestClassifier())])
    param_grid = [
        {'classifier' : [LogisticRegression(max_iter=10000, multi_class='auto')],
        'classifier__penalty' : ['l1', 'l2'],
        'classifier__C' : np.logspace(-2, 4, 6),
        'classifier__solver' : ['liblinear']
        }
    ]

    #Classification Methods
    #from neural_network import run_NN


    def reclass_data(labels,reclass_to):
        if reclass_to == 4:
            for i in range(0, labels.shape[0]):
                if (labels[i]==5 or labels[i]==7 or labels[i]==8 or labels[i]==9 or labels[i]==11 or labels[i]==12 or labels[i]==13 or labels[i]==14 or labels[i]==15 or labels[i]==3 or labels[i]==4 or labels[i]==6):
                    labels[i] = 16
            for i in range(0, labels.shape[0]):
                if(labels[i]==10):
                    labels[i] = 3
                if(labels[i]==16):
                    labels[i] = 4
        elif reclass_to == 2:
            for i in range(0, labels.shape[0]):
                if (labels[i]>=2):
                    labels[i] = 2
        return labels

    def generate_features(features):
        n_features=copy.deepcopy(features)
        z =np.concatenate((features,n_features**2),axis =1)
        z =np.concatenate((z,n_features**3),axis =1)
        z =np.concatenate((z,np.sqrt(n_features)),axis =1)
        z =np.concatenate((z,np.cbrt(n_features)),axis =1)
        return z

    #Parse a data file and return the features and labels as a numpy ndarray
    dataset, labels = di.parse_data_file('arrhythmia.csv') 

    dataset = dataset[['Age'
                    , 'Weight'
                    , 'Average QRS Duration'
                    , 'Average P-R interval'
                    , 'Average Q-T interval'
                    , 'Average P interval'
                    , 'Average T interval'
                    , 'T angle'
                    , 'P angle'
                    , 'QRST angle'
                    , 'J angle'
                    , 'QRS angle'
                    , 'Heart rate'
                    , 'V1: Amplitude of Q wave'
                    , 'V1: Amplitude of R wave'	
                    , 'V1: Amplitude of S wave' 
                    , 'V1: Amplitude of P wave'
                    , 'V1: Amplitude of T wave'
                    , 'V1: QRSA'
                    , 'V1: QRSTA'
                    , 'V2: Amplitude of Q wave'
                    , 'V2: Amplitude of R wave'	
                    , 'V2: Amplitude of S wave' 
                    , 'V2: Amplitude of P wave'
                    , 'V2: Amplitude of T wave'
                    , 'V2: QRSA'
                    , 'V2: QRSTA'
                    , 'V3: Amplitude of Q wave'
                    , 'V3: Amplitude of R wave'	
                    , 'V3: Amplitude of S wave' 
                    , 'V3: Amplitude of P wave'
                    , 'V3: Amplitude of T wave'
                    , 'V3: QRSA'
                    , 'V3: QRSTA'
                    , 'V4: Amplitude of Q wave'
                    , 'V4: Amplitude of R wave'	
                    , 'V4: Amplitude of S wave' 
                    , 'V4: Amplitude of P wave'
                    , 'V4: Amplitude of T wave'
                    , 'V4: QRSA'
                    , 'V4: QRSTA'
                    , 'V5: Amplitude of Q wave'
                    , 'V5: Amplitude of R wave'	
                    , 'V5: Amplitude of S wave' 
                    , 'V5: Amplitude of P wave'
                    , 'V5: Amplitude of T wave'
                    , 'V5: QRSA'
                    , 'V5: QRSTA'
                    , 'V5: Amplitude of Q wave'
                    , 'V5: Amplitude of R wave'	
                    , 'V5: Amplitude of S wave' 
                    , 'V5: Amplitude of P wave'
                    , 'V5: Amplitude of T wave'
                    , 'V5: QRSA'
                    , 'V5: QRSTA'
                    , 'V6: Amplitude of Q wave'
                    , 'V6: Amplitude of R wave'	
                    , 'V6: Amplitude of S wave' 
                    , 'V6: Amplitude of P wave'
                    , 'V6: Amplitude of T wave'
                    , 'V6: QRSA'
                    , 'V6: QRSTA'
                    ]]
    
    features = dataset.to_numpy()
    labels = labels.to_numpy()

    #Generate new 'hidden features'
    features = generate_features(features)
    labels = reclass_data(labels,4)

    X = np.zeros((len(dataset.columns),features.shape[0]))
    for i in np.arange(0,len(dataset.columns)):
        X[i] = features.T[i]
    X= np.array(X).T

    #Get a test and test tests at a 10/90 split. 
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2, random_state=None)

    #Neural Network Dataset
    #run_NN(X_train, X_test, Y_train, Y_test,dataset)


    #Random Forest Classifier
    # rfc_classifer = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_train, Y_train)

    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, n_jobs=-1).fit(X_train, Y_train)

    # hypo = rfc_classifer.predict(X_train)
    hypo = clf.predict(X_train)
    score = f1_score(Y_train, hypo, average='macro')
    print('Train: %F ' % (score), end='')
    hypo = clf.predict(X_test)
    # score = f1_score(Y_test, hypo, average='macro')
    score = f1_score(Y_test, hypo, average='macro')
    print('Test: %F' % (score))


    print('Best features:', clf.best_estimator_.get_params())
#Naive Bayes Classifier



#score = f1_score(Y_test, hypo, average='macro')

#print('F1_Measure: %F' % (score1))
