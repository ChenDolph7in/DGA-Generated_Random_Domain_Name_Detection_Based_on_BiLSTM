from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import time
import pickle
from sklearn.utils import shuffle
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.model_selection import train_test_split
from random import shuffle
import numpy

fp = open("./data_set/all_dataV2.pkl", 'rb')
all_data = pickle.load(fp)  # 提取数据

valid_chars = all_data["valid_chars"]
# print(valid_chars)

max_features = all_data["max_features"]  # 特征维度
maxlen = all_data["maxlen"]  # 每条训练数据的特征数量


def process_features(valid_chars, features, maxlen):
    """将特征替换为 检索字典的值 并根据最长特征长度 构造每个特征 无值填充为0"""
    X = [[valid_chars[y] for y in x] for x in features]  # 域名 根据检索字典 转换为 数字特征
    X = sequence.pad_sequences(X, maxlen=maxlen)  # 根据元素最大长度
    X = np.array(X)  # 转换为np.array
    return X


def predict():
    """
    预测函数  data为待预测数据 输入格式为 [域名1, 域名2, 域名3......]
    :param data:
    :return:
    """
    data_set = all_data["data_set"]
    shuffle(data_set)  # 打乱数据集 防止出现000000011111111情况
    features = [i[1] for i in data_set]  # 提取域名
    label = [i[0] for i in data_set]  # 提取标签
    y = [0 if x == 'benign' else 1 for x in label]  # 目标值修改为0或1
    x = process_features(valid_chars, features, maxlen)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)  # 数据集切割


    print(len(x_train))

    calculate("./model/DGA_predict_BLSTM_V7.h5",x_test,y_test)
    calculate("./model/DGA_predict_RNN_V5.h5",x_test,y_test)
    calculate("./model/DGA_predict_LSTM_V6.h5",x_test,y_test)

def calculate(model_path,x_test,y_test):
    print(model_path," : ")
    start_time = time.time()
    model = load_model(model_path)
    val_predict = (numpy.asarray(model.predict(
        x_test))).round()
    val_targ = y_test
    end_time = time.time()
    _val_accuracy = accuracy_score(val_targ,val_predict)
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    print("_val_accuracy", _val_accuracy)
    print("_val_f1", _val_f1)
    print("_val_recall", _val_recall)
    print("_val_precision", _val_precision)
    print("花费时间为{}".format(end_time - start_time))
predict()
