#!/usr/bin/env python 
# -*- coding:utf-8 -*-


from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import importlib,sys
importlib.reload(sys)
from sklearn.metrics import classification_report
from numpy import argmax, repeat
from sklearn.naive_bayes import GaussianNB
from numpy import argmax, concatenate as np_concate, repeat



#  载入数据，做预处理(分词)，切分训练集与测试集
def load_file_and_preprocessing():
    pos = pd.read_excel('/Users/leegho/Desktop/pos.xlsx', header=None, index=None)
    neg = pd.read_excel('/Users/leegho/Desktop/neg.xlsx', header=None, index=None)

    cw = lambda x: list(jieba.cut(x))

    # 新增一列 word ,存放分好词的评论，pos[0]代表表格第一列
    pos['words'] = pos[0].apply(cw)
    #cal['words'] = cal[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    # np.ones(len(pos)) 新建一个长度为len(pos)的数组并初始化元素全为1来标注好评
    # np.concatenate（）连接数组
    # axis=0 向下执行方法 axis=1向右执行方法
    labels_pos = repeat(0, 10000)
    labels_neg = repeat(1, 10000)
    y = np.concatenate((labels_pos, labels_neg), axis=0)
    y = y.astype("int32")

    # train_test_split：从样本中随机的按比例选取train data和testdata
    # 一般形式：train_test_split(train_data,train_target,test_size=0.4, random_state=0)
    # train_data：所要划分的样本特征集
    # test_size：样本占比，如果是整数的话就是样本的数量
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'],  neg['words'])), y, test_size=0.7)

    np.save('/Users/leegho/Documents/pape/bayes_data/y_train.npy', y_train)
    np.save('/Users/leegho/Documents/pape/bayes_data/y_test.npy', y_test)
    return x_train, x_test

# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    # if count != 0:
    #     vec /= count
    return vec

# 计算词向量
def get_train_vecs(x_train, x_test):
    n_dim = 300
    # 初始化模型和词表
    imdb_w2v = Word2Vec(x_train, size=n_dim, min_count=5)
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    np.save('/Users/leegho/Documents/pape/bayes_data/train_vecs.npy', train_vecs)
    print(train_vecs.shape)
    # 在测试集上训练
    imdb_w2v.train(x_test,
                   total_examples=imdb_w2v.corpus_count,
                   epochs=imdb_w2v.iter)

    imdb_w2v.save('/Users/leegho/Documents/pape/bayes_data/w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save('/Users/leegho/Documents/pape/bayes_data/test_vecs.npy', test_vecs)
    print(test_vecs.shape)

def get_data():
    train_vecs = np.load('/Users/leegho/Documents/pape/bayes_data/train_vecs.npy')
    y_train = np.load('/Users/leegho/Documents/pape/bayes_data/y_train.npy')
    test_vecs = np.load('/Users/leegho/Documents/pape/bayes_data/test_vecs.npy')
    y_test = np.load('/Users/leegho/Documents/pape/bayes_data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test

def train_bayes(train_vecs, y_train):
    model = GaussianNB()
    model.fit(train_vecs, y_train)
    joblib.dump(model, '/Users/leegho/Documents/pape/bayes_data/sentiment_bayes_model.m')

'''基于bayse分类器的预测'''
def evaluate_bayes(model_filepath, X_test, Y_test):
    model = joblib.load(model_filepath)
    Y_predict = list()
    Y_test = list(Y_test)
    right = 0
    for sent in X_test:
        Y_predict.append(model.predict(sent.reshape(1, -1))[0])
    for index in range(len(Y_predict)):
        if int(Y_predict[index]) == int(Y_test[index]):
            right += 1
    score = right / len(Y_predict)
    print('model accuray is :{0}'.format(score))
    return score

if __name__ == '__main__':
    # 加载数据
    x_train, x_test = load_file_and_preprocessing()
    # 获得词向量
    get_train_vecs(x_train, x_test)
    # 获得训练集与测试集
    train_vecs, y_train, test_vecs, y_test = get_data()
    # 训练模型
    train_bayes(train_vecs, y_train)

    clf = joblib.load('/Users/leegho/Documents/pape/bayes_data/sentiment_bayes_model.m')
    y_pred = clf.predict(test_vecs)
    print(classification_report(y_test, y_pred))
    model_filepath = '/Users/leegho/Documents/pape/bayes_data/sentiment_bayes_model.m'
    evaluate_bayes(model_filepath, test_vecs, y_test)

    # 把训练集连同测试集的预测结果结合，重新训练模型
    y_pred = clf.predict(test_vecs)
    xt = np_concate((train_vecs, test_vecs), axis=0)
    yt = np_concate((y_train, y_pred), axis=0)
    train_bayes(xt, yt)
    clf = joblib.load('/Users/leegho/Documents/pape/bayes_data/sentiment_bayes_model.m')
    y_pred = clf.predict(test_vecs)
    print(classification_report(y_test, y_pred))
    model_filepath = '/Users/leegho/Documents/pape/bayes_data/sentiment_bayes_model.m'
    evaluate_bayes(model_filepath, test_vecs, y_test)



