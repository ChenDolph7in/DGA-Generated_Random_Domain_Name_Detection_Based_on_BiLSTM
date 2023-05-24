import pandas as pd
import re
import pickle
import numpy as np
import os
from tqdm import tqdm
import time
"""
提取100万360_DGA 域名作 与100万top-1m域名 作为训练数据
提取特征与替换目标值 构造为[(目标值, 特征值), ()....] 的数据集
将重要参数 与 数据集 保存为 pki文件方便调用
"""


def get_alexa(filename):
    """
    提取alexa 域名 全部为正面域名 结构为 [[域名1, 目标值], [域名2, 目标值]..........]
    :return:
    """
    pbar = tqdm(1)
    alexa_data = pd.read_csv(filename, header=None).iloc[0:1000000, :]
    pbar.set_description('Processing ' + filename)
    time.sleep(0.2)
    # print(alexa_data)
    return [("benign", row[1]) for index, row in alexa_data.iterrows()]


def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist


def get_dgarchive(dir_name):
    list = get_filelist(dir_name, [])
    dgarchive_data = []
    pbar = tqdm(list)
    for file_name in pbar:
        new_data = get_dgarchive_DGA(file_name)
        dgarchive_data = dgarchive_data + new_data
        pbar.set_description('Processing ' + file_name)
        time.sleep(0.2)
    return dgarchive_data


def get_dgarchive_DGA(filename):
    dgarchive_data = pd.read_csv(filename, header=None).iloc[0:20000, :]
    # print(dgarchive_data)
    return [("malicious", row[0]) for index, row in dgarchive_data.iterrows()]


def get_data():
    """
    拼接返回全部数据 作为数据集
    :return:
    """
    return get_alexa("./raw_data/top-1m.csv") + get_dgarchive("./raw_data/2019-01-07-dgarchive_full")


# data_set = get_data()
# # print(data_set)
# features = [i[1] for i in data_set]   #提取域名
# label = [i[0] for i in data_set]      #提取标签
#
# valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(features)))} #构造检索字典
# max_features = len(valid_chars) + 1  #特征维度 检索字典的长度
# maxlen = np.max([len(x) for x in features])   #特征数量
#
# item = {"data_set": data_set, "valid_chars": valid_chars, "max_features": max_features, "maxlen": maxlen}
#
# #存储数据
# fp = open('./data_set/all_dataV2.pkl', 'wb')
# pickle.dump(item, fp)

##验证
fp = open("./data_set/all_dataV2.pkl", 'rb')
data = pickle.load(fp)
# print(len(data))
# print(data)
# print(data.keys())
# print(data["data_set"])
print(len(data["data_set"]))
# print(data["valid_chars"])
print(data["max_features"])
print(data["maxlen"])