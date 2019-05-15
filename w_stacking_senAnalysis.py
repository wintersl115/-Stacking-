#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import gensim
import numpy as np
from keras.models import load_model
from numpy import argmax, concatenate as np_concate, repeat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def stack_modelA(n_split, data_sets, base_models):
    second_x = []
    second_y = []

    x = data_sets[0]
    y = data_sets[1]

    k_fold = KFold(shuffle=True, n_splits=n_split)
    for train_index, _ in k_fold.split(x):
        second_x_per_model = []
        train_x, train_y = x[train_index], y[train_index]
        # test_x, test_y = x[test_index], y[test_index]

        for model in base_models:
            print(model)
            y_pred = base_models[model].predict(train_x)
            y_pred = y_pred.reshape((-1,1))
            second_x_per_model.append(y_pred)

        second_x.append(np_concate(second_x_per_model, axis=-1))
        second_y.append(train_y)

    second_x = np_concate(second_x, axis=0)
    second_y = np_concate(second_y, axis=0)

    return second_x, second_y

def stack_modelB(n_split, data_sets, base_models):
    second_X = []
    second_Y = []

    x = data_sets[0]
    y = data_sets[1]

    k_fold = KFold(shuffle=True, n_splits=n_split)
    for train_index, _ in k_fold.split(x):
        second_x_per_model = []
        train_x, train_y = x[train_index], y[train_index]
        # test_x, test_y = x[test_index], y[test_index]

        for model in base_models:
            print(model)
            y_pred = base_models[model].predict(train_x).argmax(axis=-1)
            y_pred = y_pred.reshape((-1, 1))
            second_x_per_model.append(y_pred)

        second_X.append(np_concate(second_x_per_model, axis=-1))
        second_Y.append(train_y)

    second_X = np_concate(second_X, axis=0)
    second_Y = np_concate(second_Y, axis=0)
    #meta_model.fit(second_x, second_y)
    return second_X, second_Y

def predict(test_x, meta_model, base_models):
    base_preds = []
    for model in base_models:
        pred = base_models[model].predict(test_x).argmax(axis=-1)
        pred = pred.reshape((-1, 1))
        base_preds.append(pred)
    base_pred = np_concate(base_preds, axis=-1)
    return meta_model.predict(base_pred)


if __name__ == '__main__':
    X_train = np.load('/Users/leegho/Documents/pape/bayes_data/train_vecs.npy')
    y_train = np.load('/Users/leegho/Documents/pape/bayes_data/y_train.npy')
    X_test = np.load('/Users/leegho/Documents/pape/bayes_data/test_vecs.npy')
    y_test = np.load('/Users/leegho/Documents/pape/bayes_data/y_test.npy')

    bayes_model = joblib.load('/Users/leegho/Documents/pape/bayes_data/sentiment_bayes_model.m')
    svm_model = joblib.load('/Users/leegho/Documents/pape/svm_data/model.pkl')
    second_x, second_y = stack_modelA(5, (X_train, y_train), {"bayes": bayes_model, "svm": svm_model})

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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, stratify=y)

    cnn_model = load_model('/Users/leegho/Documents/pape/cnn_model_binary.h5')
    lstm_model = load_model('/Users/leegho/Documents/pape/nlstm/sentiment_lstm_model.h5')
    second_X, second_Y = stack_modelB(5, (X_train, y_train), {"cnn": cnn_model, "lstm": lstm_model})

    fsecond_x = np_concate((second_x, second_X), axis=0)
    fsecond_y = np_concate((second_y, second_Y), axis=0)

    dt = DecisionTreeClassifier()
    meta_model = dt
    meta_model.fit(fsecond_x, fsecond_y)
    pred = predict(X_test, meta_model, {"cnn": cnn_model, "lstm": lstm_model})
    print(classification_report(y_test, pred))




