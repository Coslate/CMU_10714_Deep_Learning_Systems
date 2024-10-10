from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

def restoreShape(input_shape, input_axes):
    restore_shape = list(input_shape)
    if input_axes == None:
        restore_shape = input_axes
    else:
        if isinstance(input_axes, tuple):
            for i in input_axes:
                restore_shape[i] = 1
        else:
            restore_shape[input_axes] = 1
    return restore_shape

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        axes = 1
        Zmax = array_api.max(Z, axis=axes)
        restore_shape = restoreShape(Z.shape, input_axes=axes)
        Zmax_reshaped = array_api.reshape(Zmax, restore_shape)
        Z_minus_Zmax = Z - Zmax_reshaped
        Zexp = array_api.exp(Z_minus_Zmax)
        Zsum = array_api.sum(Zexp, axis=axes)
        Zlogsumexp = array_api.log(Zsum) + Zmax
        Zlogsumexp_reshaped = array_api.reshape(Zlogsumexp, restore_shape)
        Zres = Z - Zlogsumexp_reshaped
        return Zres
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # s = Softmax(Z)
        # gradient of logsumexp
        #s_s = multiply(s, s)
        #dsftmax = add(s, mul_scalar(s_s, -1))

        # compine
        #dlogsftmax = divide(dsftmax, s)
        #res = multiply(dlogsftmax, out_grad)

        #print(f"res.shape = {res.shape}")
        s = exp(node) 
        out_grad_sum = summation(out_grad, axes=1)
        restore_shape = restoreShape(s.shape, input_axes=1)
        out_grad_sum_reshape = reshape(out_grad_sum, restore_shape)
        out_grad_sum_bc = broadcast_to(out_grad_sum_reshape, s.shape)
        out_grad_prod = multiply(out_grad_sum_bc, s)
        res = add(out_grad, mul_scalar(out_grad_prod, -1))

        return res
        ### END YOUR SOLUTION

def logsoftmax(a):
    return LogSoftmax()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Zmax = array_api.max(Z, axis=self.axes)

        restore_shape = restoreShape(Z.shape, self.axes)
        Zmax_reshaped = array_api.reshape(Zmax, restore_shape)
        Z_minus_Zmax = Z - Zmax_reshaped
        Zexp = array_api.exp(Z_minus_Zmax)
        Zsum = array_api.sum(Zexp, axis=self.axes)
        Zres = array_api.log(Zsum) + Zmax
        return Zres
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        restore_shape = restoreShape(Z.shape, self.axes)
        l_reshape = reshape(node, restore_shape)
        out_grad_reshape = reshape(out_grad, restore_shape)
        res = exp(Z-l_reshape)
        return multiply(res, out_grad_reshape)
        ### END YOUR SOLUTION

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

