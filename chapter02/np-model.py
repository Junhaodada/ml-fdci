import numpy as np
from data import *


def linear_loss(X: np.ndarray, y, w, b):
    """线性回归模型主体
    Args:
        X: 输入变量
        y: 输出标签
        w: 变量参数权重
        b: 偏置

    Returns:
        y_hat: 预测值
        loss: 均方损失
        dw: 权重系数的一阶偏导
        db: 偏置一阶偏导
    """

    # 训练样本数量
    num_train = X.shape[0]
    # 训练特征数量
    num_feature = X.shape[1]
    # 线性回归预测输出
    y_hat = np.dot(X, w) + b
    # 计算预测输出与实际标签之间的均方损失
    loss = np.sum((y_hat - y) ** 2) / num_train
    # 基于均方损失对权重参数的一阶偏导数
    dw = np.dot(X.T, (y_hat - y)) / num_train
    # 基于均方损失对偏差项的一阶偏导数
    db = np.sum((y_hat - y)) / num_train
    return y_hat, loss, dw, db


def initialize_params(dims):
    """初始化模型参数
    Args:
        dims: 训练数据的变量维度

    Returns:
        初始化后的w和b
    """
    # 初始化权重参数为零矩阵
    w = np.zeros((dims, 1))
    # 初始化偏差参数为零
    b = 0
    return w, b


def linear_train(X: np.ndarray, y: np.ndarray, learning_rate=0.01, epochs=10000):
    """线性回归模型训练过程

    Args:
        X: 训练集X
        y: 训练集y
        learning_rate: 学习率
        epochs: 训练次数

    Returns:
        loss_his-迭代损失, params-迭代参数, grads-迭代梯度
    """
    # 记录训练损失的空列表
    loss_his = []
    # 初始化模型参数
    w, b = initialize_params(X.shape[1])
    # 迭代训练
    for i in range(1, epochs):
        # 计算当前迭代的预测值、损失和梯度
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        # 基于梯度下降的参数更新
        w += -learning_rate * dw
        b += -learning_rate * db
        # 记录当前迭代的损失
        loss_his.append(loss)
        # 每1000次迭代打印当前损失信息
        if i % 10000 == 0:
            print('epoch %d loss %f' % (i, loss))
        # 将当前迭代步优化后的参数保存到字典
        params = {
            'w': w,
            'b': b
        }
        # 将当前迭代步的梯度保存到字典
        grads = {
            'dw': dw,
            'db': db
        }
    return loss_his, params, grads


loss_his, params, grads = linear_train(X_train, y_train, 0.01, 200000)
print(params)
