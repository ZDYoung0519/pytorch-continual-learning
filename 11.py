# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:52:41 2019

@author: Administrator

本方法仅做拟合，不提取梯度，梯度是由预测数据求散度得到
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import time
import os
from scipy import interpolate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PhysicsInformedNN:
    def __init__(self):

        #  输入输出隐藏层
        self.n_input = 3
        self.n_output = 1
        self.n_hidden_1 = 64
        self.n_hidden_2 = 64
        self.n_hidden_3 = 64
        self.n_hidden_4 = 16

        self.x_train_tf = tf.placeholder(tf.float32, shape=[None, self.n_input])  # 横排 占位符
        self.y_train_tf = tf.placeholder(tf.float32, shape=[None, self.n_output])  #

        self.pred_v = self.neural_net(self.x_train_tf)

        self.loss = tf.reduce_mean(tf.square(self.pred_v - self.y_train_tf)) + 1 * tf.reduce_mean(
            tf.square(tf.sign(self.pred_v - baseline_V) - tf.sign(self.x_train_tf[:, 0] - baseline_H)))
        self.loss_2 = tf.reduce_mean(tf.square(self.pred_v - self.y_train_tf))
        learning_rate = 0.002
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # tf placeholders and graph
        cpu_num = 2
        self.sess = tf.Session(
            config=tf.ConfigProto(device_count={"CPU": cpu_num}, intra_op_parallelism_threads=cpu_num,
                                  log_device_placement=True))  # 自动分配CPU计算资源

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, x):
        self.weight_h1 = self.xavier_init(size=[self.n_input, self.n_hidden_1])
        self.biase_h1 = tf.Variable(tf.zeros([1, self.n_hidden_1]), dtype=tf.float32)
        self.weight_h2 = self.xavier_init(size=[self.n_hidden_1, self.n_hidden_2])
        self.biase_h2 = tf.Variable(tf.zeros([1, self.n_hidden_2]), dtype=tf.float32)
        self.weight_h3 = self.xavier_init(size=[self.n_hidden_2, self.n_hidden_3])
        self.biase_h3 = tf.Variable(tf.zeros([1, self.n_hidden_3]), dtype=tf.float32)
        self.weight_h4 = self.xavier_init(size=[self.n_hidden_3, self.n_hidden_4])
        self.biase_h4 = tf.Variable(tf.zeros([1, self.n_hidden_4]), dtype=tf.float32)
        self.weight_out = self.xavier_init(size=[self.n_hidden_4, self.n_output])
        self.biase_out = tf.Variable(tf.zeros([1, self.n_output]), dtype=tf.float32)

        self.layer_1 = tf.nn.relu(tf.matmul(x, self.weight_h1) + self.biase_h1)
        self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.weight_h2) + self.biase_h2)
        self.layer_3 = tf.nn.relu(tf.matmul(self.layer_2, self.weight_h3) + self.biase_h3)
        self.layer_4 = tf.nn.relu(tf.matmul(self.layer_3, self.weight_h4) + self.biase_h4)
        out_layer = tf.matmul(self.layer_4, self.weight_out) + self.biase_out

        return out_layer

    def train_U(self, train_x, train_y):

        tf_dict = {self.x_train_tf: train_x, self.y_train_tf: train_y}
        loss_record = np.array([])
        for epoch in range(800):
            for batch_data in mini_batches(train_x, train_y, mini_batch_size=256, seed=0):
                tf_dict = {self.x_train_tf: batch_data[0], self.y_train_tf: batch_data[1]}
                self.sess.run(self.optimizer, feed_dict=tf_dict)
                loss = self.sess.run(self.loss_2, feed_dict=tf_dict)
                print('Loss:', loss)
                loss_record = np.append(loss_record, loss)

        self.saver.save(self.sess, save_str + "model")
        return loss_record

    def loadnet(self):
        self.saver.restore(self.sess, save_str + "model")

    def predict_U(self, mid_x):
        V = self.sess.run(self.pred_v, {self.x_train_tf: mid_x})
        return V


# 定义函数实现mini_batch
def mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[0]  # m是样本数

    mini_batches = []  # 用来存放一个一个的mini_batch
    permutation = list(np.random.permutation(m))  # 打乱标签
    shuffle_X = X[permutation, :]  # 将打乱后的数据重新排列
    shuffle_Y = Y[permutation, :]

    num_complete_minibatches = int(m // mini_batch_size)  # 样本总数除以每个batch的样本数量
    for i in range(num_complete_minibatches):
        mini_batch_X = shuffle_X[i * mini_batch_size:(i + 1) * mini_batch_size, :]
        mini_batch_Y = shuffle_Y[i * mini_batch_size:(i + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        # 如果样本数不能被整除，取余下的部分
        mini_batch_X = shuffle_X[num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = shuffle_Y[num_complete_minibatches * mini_batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range, (0. - np.min(data)) / _range


def   main_loop(train_x, train_y):
    # test_wave=(np.array([0.034766,-0.0533111,-0.314139,5.])[:,np.newaxis].T - min_normal)/range_normal
    model_phi = PhysicsInformedNN()
    loss_all = model_phi.train_U(train_x, train_y)
    phi = model_phi.predict_U(Z[:, :-1])
    return phi, loss_all


def usenet(train_x):
    model_phi = PhysicsInformedNN()
    model_phi.loadnet()
    phi = model_phi.predict_U(Z[:, :-1])
    return phi


save_str = 'log/'
Z = np.loadtxt('all-new-for-filter.csv', delimiter=',')
# Z=np.loadtxt('all_new.csv', delimiter=',')

runtype = 0  # 0为训练，1为应用

# 运动的上界与下界？
range_normal = np.array([max(Z[:, 0]), max(Z[:, 1]), max(Z[:, 2]), max(Z[:, 3])])
min_normal = np.array([min(Z[:, 0]), min(Z[:, 1]), min(Z[:, 2]), min(Z[:, 3])])

# 标准化数据
_, baseline_H = normalization(Z[:, 0])
_, baseline_V = normalization(Z[:, -1])
for i in range(len(Z[1, :])):
    Z[:, i], _ = normalization(Z[:, i])
# print('shape:Z',end='')
# print(np.shape(Z))
# 扩列
train_X = Z[0, :-1][:, np.newaxis].T
train_Y = Z[0, -1:][:, np.newaxis].T

# print('shape:train_X', end='')
# print(np.shape(train_X))
# print('shape:train_Y', end='')
# print(np.shape(train_Y))

# mini_batches
Ztrain = mini_batches(Z[:, :-1], Z[:, -1:], mini_batch_size=256, seed=0)
# print('shape:Ztrain', end='')
# print(np.shape(Ztrain))

test_X = Ztrain[0][0]
test_Y = Ztrain[0][1]
# print('shape:test_X', end='')
# print(np.shape(test_X))
# print('shape:test_Y', end='')
# print(np.shape(test_Y))
for n_data in Ztrain[1:3]:
    test_X = np.concatenate([test_X, n_data[0]], axis=0)
    test_Y = np.concatenate([test_Y, n_data[1]], axis=0)
# print('shape:test_X', end='')
# print(np.shape(test_X))
# print('shape:test_Y', end='')
# print(np.shape(test_Y))


for n_data in Ztrain[3:]:
    train_X = np.concatenate([train_X, n_data[0]], axis=0)
    train_Y = np.concatenate([train_Y, n_data[1]], axis=0)
# print(np.shape(train_X))
# print(np.shape(train_Y))
# print('shape:train_X', end='')
# print(np.shape(train_X))


phi, loss_all = main_loop(train_X, train_Y)
# phi = usenet(train_X)

line1 = plt.plot(phi[:])
line2 = plt.plot(Z[:, -1:], linestyle='--')
# plt.legend( ['pre','act'])

# plt.plot(loss_all)
# plt.xlim(0, 300)

plt.show()

