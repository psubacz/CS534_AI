# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:43:22 2019

@author: Eastwind
"""

#Keras Modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import f1_score

def run_NN(X_train, X_test, Y_train, Y_test,dataset):
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
    score = f1_score(Y_test, hypo, average='macro')
    print('F1_Measure: %F' % (score))