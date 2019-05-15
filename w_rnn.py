#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from numpy import argmax, concatenate as np_concate, repeat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

if __name__ == '__main__':
    from sklearn.externals import joblib
    # 加载数据
    with open('/Users/leegho/Desktop/pos_embedding_tensor.pkl', 'rb') as fr:
        embedding_tensor_pos = joblib.load(fr)
    labels_pos = repeat(0, 10000)
    with open('/Users/leegho/Desktop/neg_embedding_tensor.pkl', 'rb') as fr:
        embedding_tensor_neg = joblib.load(fr)
    labels_neg = repeat(1, 10000)

    x = np_concate((embedding_tensor_pos, embedding_tensor_neg), axis=0)
    x = x.astype("float32")
    y = np_concate((labels_pos,labels_neg), axis=0)
    y = y.astype("int32")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Dropout(0.5))
    model.add(SimpleRNN(32))
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation="sigmoid"))
    #print(model.summary())
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # 训练模型
    model.fit(X_train, y_train, epochs = 10, batch_size = 32,  verbose = True)
    model.save('/Users/leegho/Documents/pape/rnn/sentiment_rnn_model.h5')
    model.evaluate(X_test, y_test, verbose=True)

    y_pred = model.predict(X_test)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    print(classification_report(y_test, y_pred))

    # 把训练集连同测试集的预测结果结合，重新训练模型
    y_pred = model.predict(X_test)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    xt = np_concate((X_train, X_test), axis=0)
    yt = np_concate((y_train, y_pred), axis=0)
    model.fit(xt, yt, batch_size=32, epochs=10)
    model.save('/Users/leegho/Documents/pape/rnn/sentiment_rnn_model.h5')
    model.evaluate(X_test, y_test, verbose=True, batch_size=32)
    y_pred = model.predict(X_test)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    print(classification_report(y_test, y_pred))

