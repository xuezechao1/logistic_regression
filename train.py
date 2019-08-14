#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from xzc_tools import tools
from matplotlib import pyplot as plt

@tools.funcRunTime
def load_data(file_path):
    """
    导入训练数据
    :param file_path:数据绝对路径
    :return: 返回特征，标签
    """
    try:
        feature_data = []
        label_data = []
        with open(file_path) as f:
            for line in f:
                feature_temp = []
                label_temp = []
                line = line.strip().split('\t')
                line = [float(i) for i in line]
                feature_temp.append(1)
                feature_temp.extend(line[0:2])
                label_temp.append(line[-1])
                feature_data.append(feature_temp)
                label_data.append(label_temp)
        return np.mat(feature_data), np.mat(label_data)
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

def sigmoid(x):
    try:
        return 1.0 / (1 + np.exp(-x))
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

def error_rate(sigmoid_value, label):
    try:
        data_num = np.shape(sigmoid_value)[0]
        sum_err = 0.0
        for i in range(data_num):
            if 0 < sigmoid_value[i, 0] < 1:
                sum_err -= (label[i, 0] * np.log(sigmoid_value[i, 0]) +
                            (1 - label[i, 0]) * np.log(1-sigmoid_value[i, 0]))
            else:
                sum_err -= 0
        return float(sum_err / data_num)
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

@tools.funcRunTime
def train_data(feature, label, maxCycle, alpha):
    try:
        feature_num = np.shape(feature)[1]
        w = np.mat(np.ones((feature_num, 1)))
        i = 0
        while i < maxCycle:
            i += 1
            sigmoid_value = sigmoid(feature * w)
            err = label - sigmoid_value
            if i % 100 == 0:
                tools.printInfo(1, '训练次数:{0},错误率:{1}'.format(i,error_rate(sigmoid_value, label)))
            w = w + alpha * feature.T * err
        return w
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

@tools.funcRunTime
def save_model(w):
    try:
        w = w.tolist()
        w = [str(w[0][0]), str(w[1][0]), str(w[2][0])]
        tools.writeFile(2, 'w_info.txt', '\t'.join(w))
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

@tools.funcRunTime
def show_model(feature, w):
    try:
        w = w.tolist()
        feature_x = [float(i[0]) for i in feature[:,1]]
        feature_y = [float(i[0]) for i in feature[:,2]]
        w_y = [(w[0][0] - w[1][0] * i) / w[2][0] for i in feature_x]

        plt.title('logistic regression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(feature_x, feature_y, 'ob')
        plt.plot(feature_x, w_y, 'r')
        plt.show()

    except Exception as msg:
        tools.printInfo(2, msg)

if __name__ == '__main__':
    # 导入数据
    tools.printInfo(1, '导入训练数据:')
    data_path = os.path.abspath(sys.argv[1])
    feature, label = load_data(data_path)

    # 训练数据
    maxCycle = int(input('请输入最大循环次数(比0大):\n'))
    alpha = float(input('请输入学习率(0, 1):\n'))
    if maxCycle <= 0:
        tools.printInfo(3, '最大循环次数数值错误，请重新运行并输入正确的值!')
        sys.exit()
    if alpha <=0 or alpha >= 1:
        tools.printInfo(3, '学习率数值错误，请重新运行并输入正确的值!')
        sys.exit()
    tools.printInfo(1, '最大循环次数和学习率符合范围，开始训练数据:')
    w = train_data(feature, label, maxCycle, alpha)

    # 保存训练结果
    tools.printInfo(1, '训练结束，保存训练结果')
    save_model(w)

    # 画图显示结果
    show_model(feature, w)