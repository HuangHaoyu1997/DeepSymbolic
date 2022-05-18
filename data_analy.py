import pickle, torch, gym
import numpy as np
from core.utils import print_matrix
from core.DeepSymbol_v3 import DeepSymbol
from core.function import func_set
from env.CartPoleContinuous import CartPoleContinuousEnv
# import sys
# sys.path.append(r'C:/Users/44670/Documents/GitHub/DeepSymbolic')

# ckpt_deepsymbol-v3_LunarLander-v2
with open('./results/TitanXP/cartpole-v1/CMA_ES-1370.pkl', 'rb') as f:
    best_solution = pickle.load(f)
    # print(best_solution[1])

# env = wrapper(gym.make('LunarLander-v2'))
env = gym.make('CartPole-v1')
inpt_dim = env.observation_space.shape[0]
out_dim = env.action_space.n


ds = DeepSymbol(inpt_dim, out_dim, func_set)
ds.model.set_params(torch.tensor(best_solution[1][:ds.model.num_params]))
ds.fc.set_params(best_solution[1][ds.model.num_params:])
print(ds.fc.weight, ds.fc.bias)

mat1, mat2, mat3 = ds.model()
mat1_value, mat1_idx = torch.max(mat1, dim=-1)
mat2_value, mat2_idx = torch.max(mat2, dim=-1)
mat3_value, mat3_idx = torch.max(mat3, dim=-1)
mat = [mat1_idx, mat2_idx, mat3_idx]
# print(mat1_idx, mat2_idx, mat3_idx)

symbols = ['+', '-', 'x', '/', 's', 'c', '=', '0']
print_matrix(mat, symbols, 4)
# result = ds.execute_symbol_mat([1,2,3,4,5,6,7,8], [mat1_idx])
# print(result.sum(1))


idxs, _, _ = ds.sym_mat(test=True)
zero_number = (idxs[0]==7).sum()+(idxs[1]==7).sum()+(idxs[2]==7).sum()
print(zero_number.item())


from core.utils import softmax

'''[-4.6632e-03,  0,  1,  0],
[-214.44,  1, -4.6632e-03,  0],
[ 0, -215.44,  0,  0.5403]
        [0.99330715 0.00669285]'''
def policy_SM(s=[-1.0434700e-5,  3.8075376e-2, -5.9733714e-4, -4.2141248e-2]):
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

rrr = []
idxs, _, _ = ds.sym_mat(True)
for i in range(100):
    s = env.reset()
    rr = 0
    done = False
    while not done:
        # print('xxx')
        # print('state:',s)
        # action = ds.select_action(idxs, s)
        action = policy_SM(s)
        # print('yyy')
        s,r,done,_ = env.step(action)
        rr += r
    rrr.append(rr)
    # print(rr)
rrr = np.array(rrr)
print(rrr.mean(), rrr.std())