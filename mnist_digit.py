import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("D:\DataSets\MNIST_DIGIT//train.csv")
train_label = train_data.pop('label')
test_data = pd.read_csv("D:\DataSets\MNIST_DIGIT//test.csv")

X_train , X_test , y_train , y_test = train_test_split(train_data,train_label, test_size = 0.1 , random_state = 99)

X_train=X_train.values.astype('float32')
X_test=X_test.values.astype('float32')
test_data=test_data.values.astype('float32')

X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test=X_test.reshape(X_test.shape[0], 28, 28, 1)
test_data=test_data.reshape(test_data.shape[0] , 28 , 28 , 1)

X_train=X_train/255
X_test=X_test/255
test_data=test_data/255


input_shape=X_train[0].shape
print("input_shape")

model=keras.models.Sequential()
# layers
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# simple ANN now

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history=model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test,y_test))
loss, accuracy=model.evaluate(X_test,y_test)
print(accuracy)

y_pred = model.predict_classes(test_data)

class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

results = pd.Series(y_pred,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)