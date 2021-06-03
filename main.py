import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from layers import *
from model import MLP

import os

# データを用意
X, Y = fetch_openml('mnist_784', version=1, data_home="./data/", return_X_y=True)

X = (X / 255.).to_numpy()
Y = Y.to_numpy().astype("int")

# 訓練・テストデータに分割
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=2)
train_y = np.eye(10)[train_y].astype(np.int32)
test_y = np.eye(10)[test_y].astype(np.int32)
train_n = train_x.shape[0]
test_n = test_x.shape[0]

# モデルの構築
in_dim = 784 # 入力の次元
hid1_dim = 1024 # 一つ目の隠れ層の次元
hid2_dim = 512 # 二つ目の隠れ層の次元
out_dim = 10 # 出力の次元
use_dropout = True # ドロップアウトを行うかどうか
dropout_ratio = 0.2 # ドロップアウトの割合
lr = 1 # 学習率
optimizer = SGD(lr) # オプティマイザ
# optimizer = Adam(0.0005)
model = MLP([Linear(  in_dim, hid1_dim, ReLU, use_dropout=use_dropout, dropout_ratio=dropout_ratio),
             Linear(hid1_dim, hid2_dim, ReLU, use_dropout=use_dropout, dropout_ratio=dropout_ratio),
             Linear(hid2_dim,  out_dim, Softmax)], optimizer)

# 学習
n_epoch = 100 # エポックの数
batchsize = 100 # バッチサイズ

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(n_epoch):
    print('epoch %d | ' % (epoch+1), end="")

    # 訓練
    sum_loss = 0
    pred_y = []
    perm = np.random.permutation(train_n)

    for i in range(0, train_n, batchsize):
        x = train_x[perm[i: i+batchsize]]
        t = train_y[perm[i: i+batchsize]]
        sum_loss += model.train(x, t) * len(x)
        # model.y には， (N, 10)の形で，画像が0~9の各数字のどれに分類されるかの事後確率が入っている
        # そこで，最も大きい値をもつインデックスを取得することで，識別結果を得ることができる
        pred_y.extend(np.argmax(model.y, axis=1))

    loss = sum_loss / train_n

    # accuracy : 予測結果を1-hot表現に変換し，正解との要素積の和を取ることで，正解数を計算できる．
    accuracy = np.sum(np.eye(10)[pred_y] * train_y[perm]) / train_n
    train_loss += [loss]
    train_acc += [accuracy]
    print('Train loss %.3f, accuracy %.4f | ' %(loss, accuracy), end="")

    # テスト
    sum_loss = 0
    pred_y = []

    for i in range(0, test_n, batchsize):
        x = test_x[i: i+batchsize]
        t = test_y[i: i+batchsize]

        sum_loss += model.test(x, t) * len(x)
        pred_y.extend(np.argmax(model.y, axis=1))

    loss = sum_loss / test_n
    accuracy = np.sum(np.eye(10)[pred_y] * test_y) / test_n
    test_loss += [loss]
    test_acc += [accuracy]
    print('Test loss %.3f, accuracy %.4f' %(loss, accuracy))

epochs = np.arange(1, n_epoch+1)

if not os.path.isdir('graph'):
    os.mkdir('graph')

plt.figure()
plt.plot(epochs, train_loss, label="train", color='tab:blue')
am = np.argmin(train_loss)
plt.plot(epochs[am], train_loss[am], color='tab:blue', marker='x')
plt.text(epochs[am], train_loss[am]-0.005, '%.3f' % train_loss[am], horizontalalignment="center", verticalalignment="top")

plt.plot(epochs, test_loss, label="test", color='tab:orange')
am = np.argmin(test_loss)
plt.plot(epochs[am], test_loss[am], color='tab:orange', marker='x')
plt.text(epochs[am], test_loss[am]+0.005, '%.3f' % test_loss[am], horizontalalignment="center", verticalalignment="bottom")

plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig('./graph/loss.png')


plt.figure()
plt.plot(epochs, train_acc, label="train")
am = np.argmax(train_acc)
plt.plot(epochs[am], train_acc[am], color='tab:blue', marker='x')
plt.text(epochs[am], train_acc[am]+0.001, '%.4f' % train_acc[am], horizontalalignment="center", verticalalignment="bottom")

plt.plot(epochs, test_acc, label="test")
am = np.argmax(test_acc)
plt.plot(epochs[am], test_acc[am], color='tab:orange', marker='x')
plt.text(epochs[am], test_acc[am]-0.001, '%.4f' % test_acc[am], horizontalalignment="center", verticalalignment="top")

plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig('./graph/accuracy.png')