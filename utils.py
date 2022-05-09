import numpy as np


def softmax(x):
    return np.exp(x)/(np.exp(x)).sum()

def tanh(x, alpha=1.0, with_clip=100):
    '''
    带有缩放因子alpha的tanh函数
    tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
    原函数的不饱和区间太窄,引入alpha<1对x进行缩放可以扩大不饱和区间
    '''
    x = np.clip(x, -with_clip, with_clip)
    return (np.exp(alpha*x)-np.exp(-alpha*x)) / (np.exp(alpha*x)+np.exp(-alpha*x))

def sigmoid(x, alpha=1.0, with_clip=100):
    x = np.clip(x, -with_clip, with_clip)
    out = 1 / (1 + np.exp(-alpha*x))
    return out


if __name__ == "__main__":
    pass