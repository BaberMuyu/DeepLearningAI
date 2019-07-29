import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
import numpy as np

import kt_utils

batch_size = 128
num_classes = 10
num_epochs = 1

#(train_x, train_y), (test_x, test_y) = mnist.load_data()
# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# train_x = train_x.reshape([train_x.shape[0], -1])
# test_x = test_x.reshape([test_x.shape[0], -1])

# train_x = train_x.reshape([*train_x.shape, 1])
# test_x = test_x.reshape([*test_x.shape, 1])

# train_y = keras.utils.to_categorical(train_y, num_classes)
# test_y = keras.utils.to_categorical(test_y, num_classes)

# model = Sequential(
#     [Dense(units=20,activation="sigmoid", input_dim=784),
#      Dense(units=20, activation="sigmoid"),
#      Dense(units=10, activation="softmax")]
# )

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

# Normalize image vectors
train_x = X_train_orig/255.
test_x = X_test_orig/255.

# Reshape
train_y = Y_train_orig.T
test_y = Y_test_orig.T

num_classes = classes
model = Sequential(
    [Conv2D(filters=8, kernel_size=[5, 5], strides=[1,1], padding="same", activation="relu"),
     MaxPooling2D(pool_size=(2,2), strides=(2,2)),
     Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu"),
     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation="relu"),
     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     Flatten(),
     Dense(30, activation="relu"),
     Dense(1, activation="sigmoid")
     ]
)
model = Sequential()


model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(test_x, test_y))
