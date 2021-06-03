import numpy as np

# Sigmoid 関数
class Sigmoid:
    def __init__(self):
        self.y = None

    def __call__(self, x):
        y = 1 / (1 + np.exp(-x)) # 順伝播計算
        self.y = y
        return y

    def backward(self):
        return self.y * (1 - self.y) # 逆伝播計算

# ReLU 関数
class ReLU:
    def __init__(self):
        self.x = None

    def __call__(self, x):
        self.x = x
        return x * (x > 0) # 順伝播計算

    def backward(self):
        return self.x > 0 # 逆伝播計算

# Softmax 関数
class Softmax:
    def __init__(self):
        self.y = None

    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.y = y
        return y

# Dropout
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def __call__(self, x, train_flag=True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

# 線形層
class Linear:
    def __init__(self, in_dim, out_dim, activation, use_dropout=False, dropout_ratio=0):
        # He の初期値
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2) / np.sqrt(in_dim)
        self.b = np.zeros(out_dim)
        self.activation = activation()
        self.delta = None
        self.x = None
        self.dW = None
        self.db = None
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = Dropout(dropout_ratio)

    def __call__(self, x, train_flag=False):
        # 順伝播計算
        self.x = x
        u = x.dot(self.W) + self.b
        self.z = self.activation(u)
        if self.use_dropout:
            self.z = self.dropout(self.z, train_flag=train_flag)
        return self.z

    def backward(self, dout):
        # 誤差計算
        if self.use_dropout:
            dout = self.dropout.backward(dout)
        self.delta = dout * self.activation.backward()
        dout = self.delta.dot(self.W.T)

        # 勾配計算
        self.dW = self.x.T.dot(self.delta)
        self.db = np.sum(self.delta, axis=0)

        return dout

# 確率的勾配降下法
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.dW # 各層の重みを更新
            layer.b -= self.lr * layer.db # 各層のバイアスを更新

# ADAM
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def __call__(self, layers):
        if self.m is None:
            self.m, self.v = {'W': {}, 'b': {}}, {'W': {}, 'b': {}}
            for i, layer in enumerate(layers):
                self.m['W'][i] = np.zeros_like(layer.W)
                self.v['W'][i] = np.zeros_like(layer.W)
                self.m['b'][i] = np.zeros_like(layer.b)
                self.v['b'][i] = np.zeros_like(layer.b)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i, layer in enumerate(layers):
            self.m['W'][i] += (1 - self.beta1) * (layer.dW - self.m['W'][i])
            self.v['W'][i] += (1 - self.beta2) * (layer.dW**2 - self.v['W'][i])

            layer.W -= lr_t * self.m['W'][i] / (np.sqrt(self.v['W'][i]) + 1e-7)

            self.m['b'][i] += (1 - self.beta1) * (layer.db - self.m['b'][i])
            self.v['b'][i] += (1 - self.beta2) * (layer.db**2 - self.v['b'][i])

            layer.b -= lr_t * self.m['b'][i] / (np.sqrt(self.v['b'][i]) + 1e-7)
