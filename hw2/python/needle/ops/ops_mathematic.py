"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


def reduceSameShape(var, dvar):
    original_ndim = len(var.shape)
    summed = dvar
    for i in range(original_ndim):
        if var.shape[i] != dvar.shape[i]:
            summed = summed.sum(axes=i)
    return summed.reshape(var.shape)

def setToOriginalShape(var, dvar):
    original_ndim = len(var.shape)
    new_ndim      = len(dvar.shape)

    extra_dims = new_ndim - original_ndim
    if extra_dims > 0:
        summed = dvar.sum(axes=tuple(range(extra_dims)))
        dvar = reduceSameShape(var, summed)
        return dvar.reshape(var.shape)
    else:
        dvar = reduceSameShape(var, dvar)
        return dvar

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


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        #raise NotImplementedError()
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        b = node.inputs[1]
        return multiply(out_grad, multiply(b, power(a, b-Tensor(1)))), multiply(power(a, b), log(a))
        #raise NotImplementedError()
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad*self.scalar*power_scalar(a, self.scalar-1)
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        b = node.inputs[1]

        return out_grad*divide(Tensor(1), b), out_grad*divide(multiply(a, Tensor(-1)), power_scalar(b, 2))
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*1/self.scalar
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        a_dim_length = len(a.shape)
        axes_length = len(self.axes) if self.axes is not None else 0
        if self.axes is None:
            new_axes = tuple(x for x in range(a_dim_length-2)) + (a_dim_length-1, a_dim_length-2)
        else:
            if axes_length < a_dim_length:
                new_axes = [x for x in range(a_dim_length)]
                new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
                new_axes = tuple(new_axes)
            else:
                new_axes = self.axes[0:a_dim_length]
                new_axes = new_axes[::-1]
        return array_api.transpose(a, new_axes)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)

        #raise NotImplementedError()
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if out_grad.shape != a.shape:
            out_grad = setToOriginalShape(a, out_grad)

        return out_grad
        #raise NotImplementedError()
        ### END YOUR SOLUTION

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        restore_shape = list(out_grad.shape)
        if self.axes == None:
            restore_shape = tuple(1 for x in a.shape)
        else:
            if isinstance(self.axes, tuple):
                for i in self.axes:
                    restore_shape.insert(i, 1)
            else:
                restore_shape.insert(self.axes, 1)

        out_grad = out_grad.reshape(restore_shape)
        result = broadcast_to(out_grad, a.shape)

        return result
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        da = matmul(out_grad, transpose(b))
        db = matmul(transpose(a), out_grad)

        if da.shape != a.shape:
            da = setToOriginalShape(a, da)

        if db.shape != b.shape:
            db = setToOriginalShape(b, db)

        return da, db
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return mul_scalar(out_grad, -1)
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, divide(Tensor(1), node.inputs[0]))
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, exp(node.inputs[0]))
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad_relu = array_api.zeros(a.shape)

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a.cached_data[i][j] > 0:
                    grad_relu[i][j] = 1
        grad_relu = Tensor(grad_relu, dtype=a.dtype)
        res = multiply(out_grad, grad_relu)
        return res
        #raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

