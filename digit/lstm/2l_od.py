#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 23:09:28 2019

@author: ahmed
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.datasets import mnist
from keras.callbacks import TensorBoard

name = '2l_od'

log = 'tb/' + name
mdl =  'models/' + name + '.h5'

bs = 100
num_classes = 10
epk = 3
tb = TensorBoard(log_dir=log, histogram_freq=0, batch_size=bs, write_graph=True, write_grads=False, embeddings_freq=0, update_freq='epoch')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

model = Sequential()

model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add( Dropout(0.2) )
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,  epochs=epk,  validation_data=(x_test, y_test), callbacks=[tb])


score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(mdl)
