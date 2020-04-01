#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 07:46:31 2020

@author: vanshika
"""
#Part-1 Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

#Categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X1 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X[:, 1])

labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X2.fit_transform(X[:,2])

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

#Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Part-2 Building the ANN !

#Importing the modules
import keras
from keras.models import Sequential
from keras.layers import Dense

#Iniatializing 
classifier = Sequential()

#Adding input and hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10 ,nb_epoch = 100)

#Predcting the values on the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





