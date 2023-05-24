# -*- coding: utf-8 -*-
# load module
import time

from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import pickle
from sklearn.utils import shuffle
from keras.preprocessing import sequence
def process_features(valid_chars, features, maxlen):
    """将特征替换为 检索字典的值 并根据最长特征长度 构造每个特征 无值填充为0"""
    X = [[valid_chars[y] for y in x] for x in features]  # 域名 根据检索字典 转换为 数字特征
    X = sequence.pad_sequences(X, maxlen=maxlen)  # 根据元素最大长度
    X = np.array(X)  # 转换为np.array
    return X

def SVM():
    fp = open("./data_set/all_dataV2.pkl", 'rb')
    all_data = pickle.load(fp)  # 提取数据

    valid_chars = all_data["valid_chars"]
    print(valid_chars)

    max_features = all_data["max_features"]  # 特征维度
    maxlen = all_data["maxlen"]  # 每条训练数据的特征数量

    data_set = all_data["data_set"]
    shuffle(data_set)
    # print(data_set)
    features = [i[1] for i in data_set]  # 提取域名
    label = [i[0] for i in data_set]  # 提取标签


    y = [0 if x == 'benign' else 1 for x in label]  # 目标值修改为0或1
    x = process_features(valid_chars, features, maxlen)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)  # 数据集切割

    start_time = time.time()
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(x_train,y_train)
    end_time = time.time()

    print("花费时间为{}".format(end_time - start_time))
    score=clf.score(x_test,y_test)
    print('svm score: ', score)

    # svm 预测测试集
    y_pred = clf.predict(x_test)
    svm_accuracy = accuracy_score(y_test, y_pred)
    svm_precision = precision_score(y_test, y_pred)
    svm_recall = recall_score(y_test, y_pred)
    svm_f1 = f1_score(y_test, y_pred)
    print("svm_accuracy : ",svm_accuracy)
    print("svm_precision : ",svm_precision)
    print("svm_recall : ", svm_recall)
    print("svm_f1 : ", svm_f1)



if __name__ == "__main__":
    SVM()