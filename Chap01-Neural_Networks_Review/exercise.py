from dataclasses import dataclass
import numpy as np
import sys

sys.path.append('..')


@dataclass
class Affine:
    W : np.ndarray
    b : np.ndarray

    def __post_init__(self):
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

@dataclass
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx



@dataclass
class SGD:
    lr : float = 0.01

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


@dataclass
class TwoLayerNet:
    input_size : int
    hidden_size : int
    output_size : int

    def __post_init__(self):
        I, H, O = self.input_size, self.hidden_size, self.output_size

        # 가충치와 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


def main():
    x = np.random.randn(10,2)
    model = TwoLayerNet(2,4,3)
    s = model.predict(x)

if __name__ == '__main__':
    main()
