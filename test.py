# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:01:31 2019

@author: Eastwind
"""
import numpy as np
##Keras Modules
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation


#def class_to_onehot(class_label):
#    Y = np.zeros((class_label.shape[0],16))
#    for i in range(0,class_label.shape[0]):
#        Y[i][int(class_label[i])-1] = 1
#    return Y
#
#def onehot_to_class(onehot_array):
#    H = np.zeros((onehot_array.shape[0],))
#    for i in range(0,onehot_array.shape[0]):
#        #Get the index of the one hot array and convert to singleton list
#        H[i] = np.where(onehot_array[i] == 1)[0]
#    return H

#
##agglo_classifer = AgglomerativeClustering(n_clusters=16).fit(X_train)
#model = Sequential()
#model.add(Dense(units=32, activation='relu', input_dim=9))
#model.add(Dense(units=16, activation='softmax'))
#model.compile(loss='categorical_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])
#
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
#
##Create a one-hot array of answer keys
#Y = class_to_onehot(Y_train)
#
## x_train and y_train are Nusmpy arrays --just like in the Scikit-Learn API.
#model.fit(X_train, Y, epochs=10000,verbose=1)
##loss_and_metrics = model.evaluate(X_train, Y, batch_size=128)
#classes = model.predict(X_train, batch_size=128)
#
##Decode onehot array
#hypo1 = onehot_to_class(np.round(classes))
#score1 = f1_score(Y_train, hypo1, average='macro')
#print('F1_Measure: %F' % (score1))