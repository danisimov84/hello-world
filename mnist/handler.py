import json
import keras
import os
import warnings

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, InputLayer
from tensorflow.keras.datasets import mnist

warnings.filterwarnings("ignore")

def build_model(X_train, Y_train):
    model = Sequential()

    input_shape = list(X_train.shape[1:]) + [1]

    model.add(InputLayer(input_shape))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())

    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.05))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.05))

    model.add(Dense(len(set(Y_train)), activation='softmax'))
    return model

def handle(req):
    print(req)
    try:
        args = json.loads(req)
    except Exception as e_message:
        print(e_message)
        args = {'epochs': 10, 'size': 1000}
    if not isinstance(args, dict):
        args = {'epochs': 10, 'size': 1000}
    size = int(args['size'])
    epochs = int(args['epochs'])

    if not 'descr' in args:
         args['descr'] = 'None'

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    (X_train, Y_train), (X_test, Y_test) = (X_train[:size], Y_train[:size]), (X_test[:size], Y_test[:size])

    model = build_model(X_train, Y_train)

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    batch_size = 128
    
    history = model.fit(X_train[:, :, :, None], Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(X_test[:, :, :, None], Y_test))
    train_metric = model.evaluate(X_train[:, :, :, None], Y_train, verbose=0)
    test_metric = model.evaluate(X_test[:, :, :, None], Y_test, verbose=0)

    return json.dumps({'train': train_metric, 'test': test_metric, 'descr': args['descr']})

