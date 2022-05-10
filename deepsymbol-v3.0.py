'''
using CMA-ES to optimize symbol matrix
'''
import argparse, math, os, sys, gym, cma
import torch
import torch.nn.utils as utils
import torch.optim as optim
from torch.distributions import Categorical, Beta
import numpy as np
from core.function import func_set

from utils import tanh
from core.models import Model
from configuration import config
from env.CartPoleContinuous import CartPoleContinuousEnv

env_name = 'CartPoleContinuous'
env = CartPoleContinuousEnv()
inpt_dim = env.observation_space.shape[0]
env.seed(config.seed)                   # 随机数种子
torch.manual_seed(config.seed)          # Gym、numpy、Pytorch都要设置随机数种子
np.random.seed(config.seed)

# model = Model(4, len(func_set))
# print(model())

class DeepSymbol():
    def __init__(self, inpt_dim, func_set, lr) -> None:
        self.inpt_dim = inpt_dim
        self.func_set = func_set
        self.dict_dim = len(func_set)
        self.model = Model(inpt_dim = self.inpt_dim, dict_dim= self.dict_dim)
        # self.model.train()
    
    def select_action(self, idxs, state):
        # state = torch.tensor(state)
        action = self.execute_symbol_mat(state, idxs)
        action = tanh(action.item(), alpha=0.05)
        # print(action,'\n')
        return action

    def sym_mat(self,):
        '''get symbol matrix for policy'''
        mat1, mat2, mat3 = self.model()
        p1 = Categorical(mat1)
        p2 = Categorical(mat2)
        p3 = Categorical(mat3)
        idx1 = p1.sample()
        idx2 = p2.sample()
        idx3 = p3.sample()
        idxs = [idx1, idx2, idx3]
        log_prob = p1.log_prob(idx1).sum() + p2.log_prob(idx2).sum() + p3.log_prob(idx3).sum()
        entropies = p1.entropy().log().sum() + p2.entropy().log().sum() + p3.entropy().log().sum()
        # print(log_prob, entropies)
        
        return idxs, log_prob, entropies
    
    def execute_symbol_mat(self, state, idxs):
        '''symbolic calculation using state vector'''
        tmp = torch.zeros((len(idxs), self.inpt_dim, self.inpt_dim), dtype=torch.float32)
        for ii, idx in enumerate(idxs):
            for i in range(self.inpt_dim):
                for j in range(self.inpt_dim):
                    arity = self.func_set[idx[i,j]].arity
                    # 第一个symbol matrix
                    if ii == 0:
                        if arity == 1: inpt = torch.tensor([state[i]])
                        elif arity == 2: inpt = torch.tensor([state[i], state[j]])
                    # 其后symbol matrix
                    elif ii > 0:
                        if arity == 1: 
                            inpt = [tmp[ii-1,:,:].sum(1)[i]]
                        elif arity == 2: 
                            inpt = [tmp[ii-1,:,:].sum(1)[i], tmp[ii-1,:,:].sum(1)[j]]
                    # print(idx[i,j], self.func_set[idx[i,j]].name, inpt)
                    tmp[ii,i,j] = self.func_set[idx[i,j]](*inpt)
        return tmp[-1,:,:].sum()
    
    def update_parameters(self, solution, fitness):
        self.es.tell(solution, fitness)

ds = DeepSymbol(inpt_dim, func_set, config.lr)
es = cma.CMAEvolutionStrategy([0.] * ds.model.num_params,
                                config.sigma_init,
                                {'popsize': config.pop_size
                                    })

dir = './results/ckpt_deepsymbol_' + env_name

if not os.path.exists(dir):    
    os.mkdir(dir)

def rollout(env, policy:DeepSymbol, num_episode=config.rollout_episode):
    reward = 0
    for epi in range(num_episode):
        done = False
        state = env.reset()
        idx, log_prob, entropy = ds.sym_mat()
        for t in range(config.num_steps):
            action = policy.select_action(idx, state)
            state, r, done, _ = env.step(np.array([action]))
            reward += r
            if done: break
    return reward / num_episode, log_prob, entropy

# training
for epi in range(config.num_episodes):
    reward, log_prob, entropy = rollout(env, ds)
    ds.update_parameters(reward, log_prob, entropy)
    if i_episode % config.ckpt_freq == 0:
        torch.save(ds.model.state_dict(), os.path.join(dir, 'VPG-'+str(epi)+'.pkl'))
    print("Episode: {}, reward: {}".format(epi, reward))
