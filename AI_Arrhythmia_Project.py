# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:30:30 2019

@author: Eastwind
"""
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import data_ingestion as di

#Scikit-Learn Modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#Keras Modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


def class_to_onehot(class_label):
    Y = np.zeros((class_label.shape[0],16))
    for i in range(0,class_label.shape[0]):
        Y[i][int(class_label[i])-1] = 1
    return Y

def onehot_to_class(onehot_array):
    H = np.zeros((onehot_array.shape[0],))
    for i in range(0,onehot_array.shape[0]):
        #Get the index of the one hot array and convert to singleton list
        H[i] = np.where(onehot_array[i] == 1)[0]
    return H

#Parse a data file and return the features and labels as a numpy ndarray
features,labels = di.parse_data_file('arrhythmia.csv') 

#Calculate missing valeus and subsitute into missing field
X = np.zeros((9,features.shape[0]))
for i in np.arange(0,9):
    X[i] = features.T[i]

X= np.array(X).T

#Get a test and test tests at a 10/90 split. 
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.6, random_state=0)

rfc_classifer = RandomForestClassifier(n_estimators=15, max_depth=16,random_state=0).fit(X_train, Y_train)

#agglo_classifer = AgglomerativeClustering(n_clusters=16).fit(X_train)
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=9))
model.add(Dense(units=16, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

#Create a one-hot array of answer keys
Y = class_to_onehot(Y_train)

# x_train and y_train are Nusmpy arrays --just like in the Scikit-Learn API.
model.fit(X_train, Y, epochs=10000,verbose=1)
#loss_and_metrics = model.evaluate(X_train, Y, batch_size=128)
classes = model.predict(X_train, batch_size=128)

#Decode onehot array
hypo = onehot_to_class(np.round(classes))

#hypo0 = rfc_classifer.predict(X_train)
#score0 = f1_score(Y_train, hypo0, average='macro')

score = f1_score(Y_train, hypo, average='macro')
print('F1_Measure: %F' % (score))
#print('F1_Measure: %F' % (score1))
