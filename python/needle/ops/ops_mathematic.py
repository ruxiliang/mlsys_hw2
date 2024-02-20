"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return a ** self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        return out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1)
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a / b
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, out_grad * (-a/(b**2))
        # END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a / self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        # END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTI
        axis1 = self.axes[0] if self.axes is not None else -2
        axis2 = self.axes[1] if self.axes is not None else -1
        return array_api.swapaxes(a, axis1, axis2)
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node):
        # BEGIN YOUR SOLUTION
        # axis1 = self.axes[0] if self.axes is not None else -2
        # axis2 = self.axes[1] if self.axes is not None else -1
        # return Tensor(array_api.swapaxes(out_grad.numpy(), axis1, axis2))
        return transpose(out_grad, axes=self.axes)
        # END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.reshape(a, newshape=self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, shape=self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        # a -> a(1),a(2) so da = da1 + da2
        input_shape = node.inputs[0].shape
        out_shape = out_grad.shape
        return out_grad.sum(tuple(list(range(len(out_shape) - len(input_shape))) + [idx + len(out_shape) - len(input_shape) for idx, cnt in enumerate(zip(input_shape, out_shape[len(out_shape) - len(input_shape):])) if cnt[0] == 1 and cnt[0] != cnt[1]])).reshape(input_shape)
        # END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        out_shape = input_shape
        if self.axes is None:
            out_shape = (1,) * len(input_shape)
        else:
            out_shape = tuple(
                [1 if idx in self.axes else elem for idx, elem in enumerate(input_shape)])
        return out_grad.reshape(out_shape).broadcast_to(input_shape)
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        a, b = node.inputs
        ag = matmul(out_grad, transpose(b))
        bg = matmul(transpose(a), out_grad)
        if len(ag.shape) > len(a.shape):
            ag = ag.sum(
                tuple([axis for axis in range(0, len(ag.shape) - len(a.shape))]))
        if len(bg.shape) > len(b.shape):
            bg = bg.sum(
                tuple([axis for axis in range(0, len(bg.shape) - len(b.shape))]))
        return ag, bg
        # END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return -a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return negate(out_grad)
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.log(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return (a > 0) * a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.realize_cached_data().copy() > 0, device=node.device, dtype=node.dtype, requires_grad=node.requires_grad)
        # END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
