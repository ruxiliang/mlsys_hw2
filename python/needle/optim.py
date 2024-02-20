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
        for param in self.params:
            param_grad = param.grad
            if param not in self.u:
                self.u[param] = (1 - self.momentum) * (param_grad.data + self.weight_decay * param.data)
            else:
                self.u[param] = self.momentum * self.u[param].data + (1 - self.momentum) * (param_grad.data + self.weight_decay * param.data)
            u = self.u[param]
            param.data = (param - self.lr * u).detach()
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
        self.t += 1
        for param in self.params:
            param_grad = param.grad
            total_grad = (param_grad + self.weight_decay * param) if self.weight_decay != 0 else param_grad
            total_grad_sq_2 = total_grad ** 2
            self.m[param] = (1 - self.beta1) * total_grad if param not in self.m else self.beta1 * self.m[param] + (1 - self.beta1) * total_grad
            self.v[param] = (1 - self.beta2) * total_grad_sq_2 if param not in self.v else self.beta2 * self.v[param] + (1 - self.beta2) * total_grad_sq_2
            u = (self.m[param] / (1 - self.beta1 ** self.t))
            v = (self.v[param] / (1 - self.beta2 ** self.t))
            param.data = (param.data - self.lr * u.data / (v.data ** 0.5 + self.eps)).detach()
        ### END YOUR SOLUTION
