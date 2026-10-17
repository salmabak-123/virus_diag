"""Optimiseurs : SGD avec momentum et Adam."""
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def step(self, params: dict, grads: dict) -> None: ...


class SGD(Optimizer):
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr, self.momentum, self.v = lr, momentum, {}

    def step(self, params, grads):
        for k in params:
            v = self.momentum * self.v.get(k, 0) - self.lr * grads[k]
            self.v[k] = v
            params[k] = params[k] + v


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
