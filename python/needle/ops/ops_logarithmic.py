from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: NDArray):
        # BEGIN YOUR SOLUTION
        input_axes = self.axes if self.axes is not None else tuple(
            range(len(Z.shape)))
        max_Z = array_api.max(Z, axis=input_axes)
        shape = Z.shape
        reshape = tuple(
            [1 if axis in input_axes else axis_num for axis, axis_num in enumerate(shape)])
        max_Z_sub = array_api.broadcast_to(max_Z.reshape(reshape), shape=shape)
        sum_logits = array_api.log(array_api.exp(
            Z - max_Z_sub).sum(axis=input_axes)) + max_Z
        return sum_logits
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0].realize_cached_data()
        input_axes = self.axes if self.axes is not None else tuple(
            range(len(Z.shape)))
        max_Z = array_api.max(Z, axis=input_axes)
        shape = Z.shape
        reshape = tuple(
            [1 if axis in input_axes else axis_num for axis, axis_num in enumerate(shape)])
        max_Z = Tensor(max_Z, device=node.device, dtype=node.dtype)
        max_Z_sub = max_Z.reshape(reshape).broadcast_to(shape)
        expsum = exp(node.inputs[0] - max_Z_sub).sum(input_axes)
        out_grad_log = out_grad / expsum
        out_grad_sum = out_grad_log.reshape(reshape).broadcast_to(shape)
        return out_grad_sum * exp(node.inputs[0] - max_Z_sub)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
