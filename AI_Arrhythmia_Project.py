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
from keras.layers import Dense, Activation, Dropout 
from keras.callbacks import EarlyStopping

import os
import tensorflow as tf
# disable GPU CUDA (its slower)
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

#Parse a data file and return the features and labels as a numpy ndarray
dataset, labels, num_classes = di.parse_data_file('arrhythmia.csv') 

# view data as numpy arrays
X = dataset.to_numpy()
labels = labels.to_numpy()

#Get a test and test tests at a 80/20 split. 
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2, random_state=None)
X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.3, random_state=None)


model = Sequential()
model.add(Dense(279, activation='relu', input_dim=len(dataset.columns)))
model.add(Dropout(rate=0.6))
model.add(Dense(150, activation='relu'))
model.add(Dropout(rate=0.6))
model.add(Dense(10, activation='relu'))
model.add(Dropout(rate=0.6))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# early stopping to prevent overfitting
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2000)

#Create a one-hot array of answer keys
Y_train = keras.utils.to_categorical(Y_train, num_classes=num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes=num_classes)
Y_validate = keras.utils.to_categorical(Y_validate, num_classes=num_classes)

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
history = model.fit(X_train, Y_train, epochs=10000, verbose=0, use_multiprocessing=True
          , workers=16, validation_data=(X_validate, Y_validate), callbacks=[es])

_, train_acc = model.evaluate(X_train, Y_train, verbose=0)
_, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# run predictions
#classes = model.predict(X_test, batch_size=64)

#Decode onehot array
#hypo = np.argmax(classes, axis=1)


#score = f1_score(Y_test, hypo, average='macro')
#print('F1_Measure: %F' % (score))
