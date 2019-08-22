#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
from xzc_tools import tools
import tensorflow as tf
import numpy as np
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

@tools.funcRunTime
def tensorflow_train(feature, label, maxCycle, alpha):
    try:

        # 定义神经网络的参数
        w = tf.Variable(tf.ones([3,1]))

        # 存放训练数据的位置
        x = tf.placeholder(tf.float32, shape=(None, 3), name='x-input')
        y_real = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

        # 定义神经网络的传播过程
        y_predict = tf.matmul(x, w)

        # 定义损失函数
        y_predict = tf.sigmoid(y_predict)

        cross_entropy = -tf.reduce_mean(y_real * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)) +
                                       (1 - y_real) * tf.log(tf.clip_by_value(1 - y_predict, 1e-10, 1.0)))

        train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)

        # 创建会话
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(maxCycle):

                sess.run(train_step, feed_dict={x: feature, y_real: label})

                if i % 100 == 0:
                    total_cross_entropy = sess.run(cross_entropy, feed_dict={x: feature, y_real: label})
                    print(i, total_cross_entropy)

            return sess.run(w)

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
        w_y = [float((w[0][0] - w[1][0] * i) / w[2][0]) for i in feature_x]

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
    if alpha <= 0 or alpha >= 1:
        tools.printInfo(3, '学习率数值错误，请重新运行并输入正确的值!')
        sys.exit()
    tools.printInfo(1, '最大循环次数和学习率符合范围，开始训练数据:')
    w = tensorflow_train(feature, label, maxCycle, alpha)
    print(w)

    # 保存训练结果
    tools.printInfo(1, '训练结束，保存训练结果')
    save_model(w)

    # 画图显示结果
    show_model(feature, w)