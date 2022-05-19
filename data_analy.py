import pickle, torch, gym
import numpy as np
from core.utils import print_matrix
from core.DeepSymbol_v3 import DeepSymbol
from core.function import func_set
from env.CartPoleContinuous import CartPoleContinuousEnv
# import sys
# sys.path.append(r'C:/Users/44670/Documents/GitHub/DeepSymbolic')

# ckpt_deepsymbol-v3_LunarLander-v2
with open('./results/3090/CMA_ES-1464.pkl', 'rb') as f:
    best_solution = pickle.load(f)
    # print(best_solution[1])

env = gym.make('LunarLander-v2')
# env = gym.make('CartPole-v1')
# env = env.unwrapped
inpt_dim = 6 # env.observation_space.shape[0]
out_dim = env.action_space.n


ds = DeepSymbol(inpt_dim, out_dim, func_set, 4)
ds.model.set_params(torch.tensor(best_solution[1][:ds.model.num_params]))
ds.fc.set_params(best_solution[1][ds.model.num_params:])
print(ds.fc.weight, ds.fc.bias)

mats, _, _ = ds.sym_mat(True)
symbols = ['+', '-', 'x', '/', 's', 'c', '=', '0']
print_matrix(mats, symbols, 4)
# result = ds.execute_symbol_mat([1,2,3,4,5,6,7,8], [mat1_idx])
# print(result.sum(1))

zero_number = np.sum([(idx==7).sum().item() for idx in mats])
print('Number of None operations:',zero_number)


rrr = []
for i in range(30):
    s = env.reset()
    rr = 0
    done = False
    while not done:
        action = ds.select_action(mats, s[:6])
        s,r,done,_ = env.step(action)
        rr += r
    rrr.append(rr)
    # print(rr)
rrr = np.array(rrr)
print(rrr.mean(), rrr.std())