import pickle, torch
import numpy as np
# ckpt_deepsymbol-v3_LunarLander-v2
with open('./results/3090/CMA_ES-150.pkl', 'rb') as f:
    best_solution = pickle.load(f)
# print(best_solution.fc.weight, best_solution.fc.bias)
# print(len(best_solution.model.get_params())/3)

mat1, mat2, mat3 = best_solution.model()
mat1_value, mat1_idx = torch.max(mat1, dim=-1)
mat2_value, mat2_idx = torch.max(mat2, dim=-1)
mat3_value, mat3_idx = torch.max(mat3, dim=-1)

# print(mat1_idx, mat2_idx, mat3_idx)

symbol = ['+', '-', 'x', '/', 's', 'c', '=', '0']
for i in range(8):
    t = ''
    for j in range(8):
        # t.append(symbol[mat1_idx[i,j].item()])
        t += symbol[mat1_idx[i,j].item()]
        t += '\t'
    print(t)
for _ in range(10):
    idxs, _, _ = best_solution.sym_mat()
    zero_number = (idxs[0]==7).sum()+(idxs[1]==7).sum()+(idxs[2]==7).sum()
    print(zero_number.item())

import gym
env = gym.make('LunarLander-v2')
rrr = []
idxs, _, _ = best_solution.sym_mat()
for i in range(20):
    
    s = env.reset()
    rr = 0
    done = False
    while not done:
        action = best_solution.select_action(idxs, s)
        s,r,done,_ = env.step(action)
        rr += r
    rrr.append(rr)
    print(rr)
rrr = np.array(rrr)
print(rrr.mean(), rrr.std())