#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#!/usr/bin/env python3
# coding: utf-8

import gensim
import numpy as np
from keras.models import load_model
from numpy import argmax, concatenate as np_concate, repeat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical


'''三层lstm进行训练，迭代20次'''
def train_lstm(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    import numpy as np
    data_dim = 300 # 对应词向量维度
    timesteps = 8 # 对应序列长度
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(16, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 16
    model.add(LSTM(16, return_sequences=True))  # returns a sequence of vectors of dimension 16
    model.add(LSTM(16))  # return a single vector of dimension 16
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test, Y_test))
    model.save('/Users/leegho/Documents/pape/nlstm/sentiment_lstm_model.h5')
    model.evaluate(X_test, y_test, verbose=True, batch_size=32)



if __name__ == '__main__':
    from sklearn.externals import joblib
    # 加载数据
    with open('/Users/leegho/Desktop/pos_embedding_tensor.pkl', 'rb') as fr:
        embedding_tensor_pos = joblib.load(fr)
    labels_pos = repeat(0, 10000)
    with open('/Users/leegho/Desktop/neg_embedding_tensor.pkl', 'rb') as fr:
        embedding_tensor_neg = joblib.load(fr)
    labels_neg = repeat(1, 10000)

    x = np_concate((embedding_tensor_pos,  embedding_tensor_neg), axis=0)
    x = x.astype("float32")
    y = np_concate((labels_pos, labels_neg), axis=0)
    y = y.astype("int32")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, stratify=y)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # 训练模型
    train_lstm(X_train, y_train, X_test, y_test)
    model_filepath = '/Users/leegho/Documents/pape/nlstm/sentiment_lstm_model.h5'
    model = load_model(model_filepath)
    model.evaluate(X_test, y_test, verbose=True, batch_size=32)
    y_pred = model.predict(X_test)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    print(classification_report(y_test, y_pred))

    # 把训练集连同测试集的预测结果结合，重新训练模型
    y_pred = model.predict(X_test)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    xt = np_concate((X_train, X_test), axis=0)
    yt = np_concate((y_train, y_pred), axis=0)
    model.fit(xt, yt, batch_size=32, epochs=10)
    model.save('/Users/leegho/Documents/pape/nlstm/sentiment_lstm_model.h5')
    model.evaluate(X_test, y_test, verbose=True, batch_size=32)
    y_pred = model.predict(X_test)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    print(classification_report(y_test, y_pred))








