import sys
import os
import numpy as np
from dataclasses import dataclass
sys.path.append(os.path.abspath(os.path.curdir))



@dataclass
class RNN:
    '''Inputting the parameters for RNN Block'''
    Wx : np.ndarray
    Wh : np.ndarray
    Wb : np.ndarray

    def __post_init__(self):
        '''Initiate parameters and gradients'''
        self.params = [self.Wx, self.Wh, self.b]
        self.grads = [np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next**2)
        db = np.sum(dt, axis = 0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wh.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


@dataclass
class TimeRNN:
    '''Collection of RNN Blocks through time'''
    Wx : np.ndarray
    Wh : np.ndarray
    Wb : np.ndarray
    stateful : bool = False

    def __post_init__(self):
        self.params = [self.Wx, self.Wh, self.b]
        self.grads = [np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)]
        self.layers = None

        self.h, self.dh = None, None

    def set_state(self, h):
        '''To save RNN layers as Instance'''
        self.h = h

    def reset_state(self):
        self.h = None


    def forward(self, xs):
        '''
        xs: T개 분량의 시계열 데이터를 모은 것
        N : 미니배치의 데이터 수
        T : 시계열 데이터의 길이
        D : 입력 데이터의 크기
        '''
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wh.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t:, ]+ dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs






