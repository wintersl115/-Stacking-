#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from keras import Model
from keras.layers import Embedding, Input, Conv1D, Dense, Dropout, MaxPool1D, Flatten, concatenate
from keras.utils import to_categorical
from numpy import argmax, concatenate as np_concate, repeat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class CNN(object):
    def __init__(self, max_len, embedding_dim, window1=3, window2=4, window3=5):
        # embedding_dim = shape(embedding_matrix)[1]

        # _input = Input(shape=(max_len, ), name="input", dtype="float32")
        # word_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len,
        #                            trainable=True, weights=[embedding_matrix])(_input)

        _input = Input(shape=(max_len, embedding_dim), name="input", dtype="float32")
        hidden = Dropout(0.5)(_input)

        conv1 = Conv1D(filters=100, kernel_size=window1, strides=1, padding="valid")(hidden)
        pool1 = MaxPool1D()(conv1)
        pool1 = Dropout(0.5)(pool1)
        flatten1 = Flatten()(pool1)

        conv2 = Conv1D(filters=100, kernel_size=window2, strides=1, padding="valid")(hidden)
        pool2 = MaxPool1D()(conv2)
        pool2 = Dropout(0.5)(pool2)
        flatten2 = Flatten()(pool2)

        conv3 = Conv1D(filters=100, kernel_size=window3, strides=1, padding="valid")(hidden)
        pool3 = MaxPool1D()(conv3)
        pool3 = Dropout(0.5)(pool3)
        flatten3 = Flatten()(pool3)

        cat_data = concatenate([flatten1, flatten2, flatten3])
        hidden = Dense(units=500, activation="relu")(cat_data)
        #hidden = Dense(units=600, activation="relu")(hidden)
        # hidden = Dense(units=600, activation="relu")(hidden)
        hidden = Dense(units=300, activation="relu")(hidden)
        # hidden = Dense(units=400, activation="relu")(hidden)
        # hidden = Dense(units=300, activation="relu")(hidden)
        # hidden = Dense(units=200, activation="relu")(hidden)
        hidden = Dense(units=100, activation="relu")(hidden)

        output = Dense(units=2, activation="softmax", name="output")(hidden)

        self.model = Model(inputs=[_input], outputs=[output])
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    def fit(self, train_texts, train_labels, epochs, batch_size, test_texts, test_labels):

        print(train_texts.dtype)
        print(train_labels.dtype)

        self.model.fit(x={"input": train_texts}, y={"output": train_labels},
                       validation_data=[test_texts, test_labels],
                       shuffle=True,
                       epochs=epochs, batch_size=batch_size)

    def predict(self, texts):
        return argmax(self.model.predict(x={"input": texts}), axis=1)


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
    y = np_concate((labels_pos, labels_neg), axis=0)
    y = y.astype("int32")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, stratify=y)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # 训练模型
    cnn = CNN(8, 300, window1=3, window2=4, window3=5)
    cnn.fit(X_train, y_train, 20, 32, X_test, y_test)
    cnn.model.save('/Users/leegho/Documents/pape/cnn_model_binary.h5')
    cnn.model.evaluate(X_test, y_test, verbose=True, batch_size=32)
    y_pred = cnn.model.predict(X_test, verbose=True, batch_size=32)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    print(classification_report(y_test, y_pred))

    # 把训练集连同测试集的预测结果结合，重新训练模型
    y_pred = cnn.model.predict(X_test, verbose=True, batch_size=32)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    xt = np_concate((X_train, X_test), axis=0)
    yt = np_concate((y_train, y_pred), axis=0)
    cnn.model.fit(xt, yt,batch_size=32,epochs=10)
    cnn.model.save('/Users/leegho/Documents/pape/cnn_model_binary.h5')
    cnn.model.evaluate(X_test, y_test, verbose=True, batch_size=32)
    y_pred = cnn.model.predict(X_test, verbose=True, batch_size=32)
    y_pred = to_categorical(argmax(y_pred, axis=-1))
    print(classification_report(y_test, y_pred))



