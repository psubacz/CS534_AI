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

#Parse a data file and return the features and labels as a numpy ndarray
dataset, labels = di.parse_data_file('arrhythmia.csv') 

dataset = dataset[['Age'
                 , 'Weight'
                 , 'Average QRS Duration'
                 , 'Average P-R interval'
                 , 'Average P interval'
                 , 'QRS angle'
                 , 'Heart rate'
                 ]]

features = dataset.to_numpy()
labels = labels.to_numpy()


X = np.zeros((len(dataset.columns),features.shape[0]))
for i in np.arange(0,len(dataset.columns)):
    X[i] = features.T[i]
X= np.array(X).T

#Get a test and test tests at a 10/90 split. 
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.1, random_state=0)

# rfc_classifer = RandomForestClassifier(n_estimators=15, max_depth=16,random_state=0).fit(X_train, Y_train)

#agglo_classifer = AgglomerativeClustering(n_clusters=16).fit(X_train)
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=len(dataset.columns)))
model.add(Dense(17, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

#Create a one-hot array of answer keys
one_hot_labels_train = keras.utils.to_categorical(Y_train, num_classes=17)
one_hot_labels_test = keras.utils.to_categorical(Y_test, num_classes=17)

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(X_train, one_hot_labels_train, epochs=5000,verbose=0)
#loss_and_metrics = model.evaluate(X_train, Y, batch_size=128)

classes = model.predict(X_test, batch_size=64)

#Decode onehot array
hypo = np.argmax(classes, axis=1)

#hypo0 = rfc_classifer.predict(X_train)
#score0 = f1_score(Y_train, hypo0, average='macro')

score = f1_score(Y_test, hypo, average='macro')
print('F1_Measure: %F' % (score))
#print('F1_Measure: %F' % (score1))
