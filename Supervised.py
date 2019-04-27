import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

early_stopping_monitor = EarlyStopping(patience=3)

data_SP = pd.read_csv(r"C:\Users\Apra Gupta\Desktop\Artificial Intelligence\AI_Proj_Redone\AI_Project_Redone\data\S&P500.csv", delimiter= ";")

data_SP = data_SP.loc[:,:'UO']
data_SP = data_SP[29:1599]
train_SP = data_SP[29:1300]
test_SP = data_SP[1300:1599]

train_X = train_SP.drop(columns=['Class','Date'])
print(train_X)
print(train_X.values[0][0])

# train_Y = to_categorical(train_SP.Class)
#
# print(train_Y[0:5])
#
# test_X = test_SP.drop(columns=['Class','Date'])
# test_Y = test_SP[['Class']]
#
# #print(test_Y)
#
#
# no_columns = train_X.shape[1]
#
# print(no_columns )
#
# # #create model
# model = Sequential()
#
# #add layers to model
# model.add(Dense(250, activation='relu', input_shape=(no_columns,)))
# model.add(Dense(2000, activation='relu'))
# model.add(Dense(2000, activation='relu'))
# model.add(Dense(2, activation = 'softmax'))
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# history = model.fit(train_X, train_Y, epochs=130, validation_split=0.2, callbacks=[early_stopping_monitor])
#
# plot_model(model, to_file='model.png')
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# test_y_predictions = model.predict(train_X)
# print(test_y_predictions)

# data_SP = pd.read_csv(r"C:\Users\Apra Gupta\Desktop\Artificial Intelligence\AI_Proj_Redone\AI_Project_Redone\data\S&P500.csv", delimiter= ";")
#
# data_SP = data_SP.loc[:,:'UO']
# data_SP = data_SP[29:1599]
# train_SP = data_SP[29:1300]
# test_SP = data_SP[1300:1599]
#
# train_X = train_SP.drop(columns=['Class','Date'])
# print(train_X)
#
# train_Y = train_SP[['Class']]
# print(train_Y)
# test_X = test_SP.drop(columns=['Class','Date'])
# test_Y = test_SP[['Class']]
#
# #print(test_Y)
#
# #create model
# model = Sequential()
#
# #get number of columns in training data
# no_columns = train_X.shape[1]
# #print(no_columns )
#
# #add layers to model
# model.add(Dense(250, activation='relu', input_shape=(no_columns,)))
# model.add(Dense(250, activation='relu'))
# model.add(Dense(250, activation='sigmoid'))
# model.add(Dense(1))
#
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
# history = model.fit(train_X, train_Y, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor])
#
# plot_model(model, to_file='model.png')
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# test_y_predictions = model.predict(train_X)
#
# print(test_y_predictions)
#
#
#
#
#
#
