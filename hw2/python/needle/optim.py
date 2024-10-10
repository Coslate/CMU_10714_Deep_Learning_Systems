"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for id, w in enumerate(self.params):
            gt = ndl.Tensor(w.grad.data, dtype=w.data.dtype) + self.weight_decay*w.data
            if id not in self.u:
                self.u[id] = ndl.Tensor(0, dtype=w.data.dtype)

            self.u[id] = self.momentum*self.u[id] + (1-self.momentum)*gt
            gt = self.u[id]
            w.data = w.data - self.lr*gt
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t+=1
        for id, w in enumerate(self.params):
            gt = ndl.Tensor(w.grad.data, dtype=w.data.dtype) + self.weight_decay*w.data
            if id not in self.m:
                self.m[id] = ndl.Tensor(0, dtype=w.data.dtype)
            if id not in self.v:
                self.v[id] = ndl.Tensor(0, dtype=w.data.dtype)

            self.m[id] = self.beta1*self.m[id] + (1-self.beta1)*gt
            self.v[id] = self.beta2*self.v[id] + (1-self.beta2)*(gt**2)
            mtbar = self.m[id]/(1-self.beta1**self.t)
            vtbar = self.v[id]/(1-self.beta2**self.t)
            w.data = w.data - self.lr*mtbar/(vtbar**0.5 + self.eps)
        ### END YOUR SOLUTION
