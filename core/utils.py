import numpy as np


def softmax(x, alpha=0.1, with_clip=100):
    x = np.clip(x, -with_clip, with_clip)
    return np.exp(alpha*x)/(np.exp(alpha*x)).sum()

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

def compute_centered_ranks(x):
    ranks = np.empty(x.shape[0], dtype=np.float32)
    # x.argsort()从小到大排列,将reward归到[-0.5,0.5]区间
    ranks[x.argsort()] = np.linspace(-0.5, 0.5, x.shape[0], dtype=np.float32)
    return ranks

def compute_weight_decay(weight_decay, model_param_list):   
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

if __name__ == "__main__":
    pass