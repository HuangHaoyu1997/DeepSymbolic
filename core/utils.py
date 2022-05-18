import numpy as np


def softmax(x, alpha=0.1, with_clip=50):
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

def compute_centered_ranks(x, clip_margin=0.5):
    ranks = np.empty(x.shape[0], dtype=np.float32)
    # x.argsort()从小到大排列,将reward归到[-0.5,0.5]区间
    ranks[x.argsort()] = np.linspace(-clip_margin, clip_margin, x.shape[0], dtype=np.float32)
    return ranks

def compute_weight_decay(weight_decay, model_param_list):   
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

def wrapper(env):
    import gym
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

def print_matrix(mats, symbols, state_dim):
    
    for idx, matrix in enumerate(mats):
        print('Matrix '+str(idx+1))
        for i in range(state_dim):
            t = ''
            for j in range(state_dim):
                # t.append(symbol[mat1_idx[i,j].item()])
                t += symbols[matrix[i,j].item()]
                t += '\t'
            print(t)
        print('\n')

def policy_SM(s):
    '''
    有效的CartPole-v1游戏策略
    '''
    from core.utils import softmax
    # A = s[3] / s[2] + (s[2] + s[2] * np.sin(s[2])) / s[3]
    A = s[1] + s[3] + np.sin(s[2])
    B = s[2]**2 + np.cos(s[0])
    C = B/A
    a1 = -10.64138864 * (C-B) + 46.53673285 * np.cos(B) + 12.73689465
    a2 = 19.24603433 * (C-B) + 30.11917319 * np.cos(B) + 22.70999363
    aa = [a1, a2]
    aa = np.maximum(aa, 0)
    p = softmax(aa, with_clip=50)
    action = np.random.choice(2, p=p)
    
    return action # [a1, a2], p, 
# print(policy_SM())
if __name__ == "__main__":
    pass