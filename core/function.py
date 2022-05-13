import numpy as np
import math, torch

class Function:
    """
    A general function
    arity: 函数的输入参数的数量
    """

    def __init__(self, f, arity, name=None):
        self.f = f
        self.arity = arity
        self.name = f.__name__ if name is None else name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

def torchInv(x):
    return torch.pow(x, -1)
def torchConst(x):
    return torch.pow(x, 0)
def torchNone(x):
    return torch.tensor(0.)
def torchProDiv(x, y):
    if torch.abs(y) <= 1e-3:
        return torch.div(x, y+1e-3)
    else:
        return torch.div(x, y)
func_set = [
    Function(torch.add, 2, 'torchAdd'),
    Function(torch.sub, 2, 'torchSub'),
    Function(torch.mul, 2, 'torchMul'),
    Function(torchProDiv, 2, 'torchDiv'),
    # Function(torch.div, 2, 'torchDiv'),
    # Function(torch.max, 2, 'torchMax'),
    # Function(torch.min, 2, 'torchMin'),

    # Function(torch.log, 1, 'torchLog'),
    Function(torch.sin, 1, 'torchSin'),
    Function(torch.cos, 1, 'torchCos'),
    # Function(torch.exp, 1, 'torchExp'),
    Function(torch.neg, 1, 'torchNeg'),
    # Function(torch.abs, 1, 'torchAbs'),
    # Function(torch.square, 1, 'torchX^2'),
    # Function(torch.sqrt, 1, 'torchSqrt'),
    # Function(torch.sign, 1, 'torchSgn'),
    # Function(torch.relu, 1, 'torchRelu'),
    # Function(torchInv, 1, 'torchInv'),
    # Function(torchConst, 1, 'torchConst'),
    Function(torchNone, 1, 'torchNone'),
]
if __name__ == '__main__':
    
    pass