import pickle, torch
import numpy as np
from env.CartPoleContinuous import CartPoleContinuousEnv
import sys
sys.path.append(r'C:/Users/44670/Documents/GitHub/DeepSymbolic')
# ckpt_deepsymbol-v3_LunarLander-v2
with open('./results/TitanXP/CMA_ES-666.pkl', 'rb') as f:
    best_solution = pickle.load(f)
print(best_solution.fc.weight, best_solution.fc.bias)
# print(len(best_solution.model.get_params())/3)

mat1, mat2, mat3 = best_solution.model()
mat1_value, mat1_idx = torch.max(mat1, dim=-1)
mat2_value, mat2_idx = torch.max(mat2, dim=-1)
mat3_value, mat3_idx = torch.max(mat3, dim=-1)
mat = [mat1_idx, mat2_idx, mat3_idx]
# print(mat1_idx, mat2_idx, mat3_idx)

symbol = ['+', '-', 'x', '/', 's', 'c', '=', '0']
for matrix in mat:
    print('Matrix')
    for i in range(4):
        t = ''
        for j in range(4):
            # t.append(symbol[mat1_idx[i,j].item()])
            t += symbol[matrix[i,j].item()]
            t += '\t'
        print(t)
    print('\n')
# result = best_solution.execute_symbol_mat([1,2,3,4,5,6,7,8], [mat1_idx])
# print(result.sum(1))
'''
for _ in range(10):
    idxs, _, _ = best_solution.sym_mat(test=True)
    zero_number = (idxs[0]==7).sum()+(idxs[1]==7).sum()+(idxs[2]==7).sum()
    print(zero_number.item())

'''


from core.utils import softmax
def policy_SM(s):
    A = s[3] / s[2] + (s[2] + s[2] * np.sin(s[2])) / s[3]
    a1 = 3.7394824 * A - 11.637533
    a2 = 14.541853 * A - 0.35463876
    p = softmax([a1,a2])
    action = np.random.choice(2, p=p)
    
    return [a1, a2], p, action

    
'''

'''
import gym
def wrapper(env):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

env = wrapper(gym.make('LunarLander-v2'))
env = gym.make('CartPole-v1')
rrr = []
idxs, _, _ = best_solution.sym_mat(True)
for i in range(2):
    s = env.reset()
    rr = 0
    done = False
    while not done:
        print('xxx')
        print('state:',s)
        action = best_solution.select_action(idxs, s)
        print('SM:', policy_SM(s))
        print('yyy')
        s,r,done,_ = env.step(action)
        rr += r
    rrr.append(rr)
    print(rr)
rrr = np.array(rrr)
print(rrr.mean(), rrr.std())