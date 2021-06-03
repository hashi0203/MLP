import numpy as np

# 多層パーセプトロン
class MLP():
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def train(self, x, t):
        # 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y, train_flag=True)

        # 損失関数 (cross-entropy) の計算
        self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)

        # 誤差逆伝播
        batchsize = len(self.layers[-1].x)
        delta = (self.y - t) / batchsize
        self.layers[-1].delta = delta
        self.layers[-1].dW = np.dot(self.layers[-1].x.T, self.layers[-1].delta)
        self.layers[-1].db = np.dot(np.ones(batchsize), self.layers[-1].delta)
        dout = np.dot(self.layers[-1].delta, self.layers[-1].W.T)

        # 中間層
        for layer in self.layers[-2::-1]:
            # 中間層の誤差・勾配計算
            dout = layer.backward(dout)

        # パラメータの更新
        self.optimizer(self.layers)
        return self.loss

    def test(self, x, t):
        # 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)

        # 損失関数 (cross-entropy) の計算
        self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        return self.loss
