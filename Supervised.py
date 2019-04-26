import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=3)


data_SP = pd.read_csv(r"C:\Users\Apra Gupta\Desktop\Artificial Intelligence\AI_Proj_Redone\AI_Project_Redone\data\S&P500.csv", delimiter= ";")

data_SP = data_SP.loc[:,:'UO']
data_SP = data_SP[29:1599]
train_SP = data_SP[29:1300]
test_SP = data_SP[1300:1599]

train_X = train_SP.drop(columns=['Class','Date'])


train_Y = train_SP[['Class']]
train_Y = to_categorical(train_SP.Class)
#print(train_Y)
test_X = test_SP.drop(columns=['Class','Date'])
test_Y = test_SP[['Class']]
test_Y = to_categorical(test_SP.Class)
#print(test_Y)

#create model
model = Sequential()

#get number of columns in training data
no_columns = train_X.shape[1]

#add layers to model
model.add(Dense(250, activation='relu', input_shape=(no_columns,)))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=130, validation_split=0.2, callbacks=[early_stopping_monitor])
test_y_predictions = model.predict(test_X)

print(test_y_predictions)
print(test_Y)





