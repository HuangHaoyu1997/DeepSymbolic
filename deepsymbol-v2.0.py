import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.distributions import Categorical, Beta
import numpy as np
from function import func_set
import argparse, math, os, sys, gym, torch
from function import *
from utils import tanh
from configuration import config
from CartPoleContinuous import CartPoleContinuousEnv
import torch.nn.functional as F
import torch.optim as optim

env_name = 'CartPoleContinuous'
env = CartPoleContinuousEnv()

env.seed(config.seed)                   # 随机数种子
torch.manual_seed(config.seed)          # Gym、numpy、Pytorch都要设置随机数种子
np.random.seed(config.seed)


class Model(nn.Module):
    def __init__(self, inpt_dim, dict_dim, ):
        super(Model, self).__init__()
        self.inpt_dim = inpt_dim 
        self.dict_dim = dict_dim
        self.fc1 = nn.Linear(1, inpt_dim*inpt_dim*self.dict_dim)
        self.fc2 = nn.Linear(1, inpt_dim*inpt_dim*self.dict_dim)
        self.fc3 = nn.Linear(1, inpt_dim*inpt_dim*self.dict_dim)

    def forward(self, ):
        mat1 = self.fc1(torch.tensor([0.]))
        mat1 = mat1.view(self.inpt_dim, self.inpt_dim, self.dict_dim)
        mat1 = F.softmax(mat1, dim=-1) # mat1.shape=(4,4,5)

        mat2 = self.fc2(torch.tensor([0.]))
        mat2 = mat2.view(self.inpt_dim, self.inpt_dim, self.dict_dim)
        mat2 = F.softmax(mat2, dim=-1) # mat2.shape=(4,4,5)

        mat3 = self.fc3(torch.tensor([0.]))
        mat3 = mat3.view(self.inpt_dim, self.inpt_dim, self.dict_dim)
        mat3 = F.softmax(mat3, dim=-1) # mat3.shape=(4,4,5)
        return mat1, mat2, mat3
import cma
cma.CMAEvolutionStrategy()
model = Model(4, len(func_set))
print(model())

class DeepSymbol():
    def __init__(self, inpt_dim, func_set, lr) -> None:
        self.inpt_dim = inpt_dim
        self.func_set = func_set
        self.dict_dim = len(func_set)

        self.model = Model(inpt_dim = self.inpt_dim,
                            dict_dim= self.dict_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
    
    def select_action(self, idxs, state):
        # state = torch.tensor(state)
        # idx, log_prob, entropy = self.sym_mat(state)
        action = self.execute_symbol_mat(state, idxs)
        action = tanh(action.item(), alpha=0.1)
        return action

    def sym_mat(self,):
        '''get symbol matrix for policy'''
        matrix_prob = self.model()
        
        # find the max prob symbol for each position in matrix
        idx = torch.argmax(matrix_prob, dim=-1)
        
        # joint log prob for all symbols selected
        log_prob = 0
        for i in range(idx.size()[0]):
            for j in range(idx.size()[1]):
                log_prob += matrix_prob[i, j, idx[i,j]].log()

        # upper bound of entropy for joint symbols categorical distribution
        entropies = 0
        for i in range(idx.size()[0]):
            for j in range(idx.size()[1]):
                p = matrix_prob[i, j]
                dist = Categorical(p)
                entropy = dist.entropy()
                entropies += entropy
        
        return idx, log_prob, entropies
    
    def execute_symbol_mat(self, state, idx):
        '''
        do the symbolic calculation using state vector
        '''
        tmp = torch.zeros_like(idx, dtype=torch.float32)
        for i in range(idx.size()[0]):
            for j in range(idx.size()[1]):
                arity = self.func_set[idx[i,j]].arity
                if arity == 1:
                    inpt = torch.tensor([state[i]])
                elif arity == 2:
                    inpt = torch.tensor([state[i], state[j]])
                # print(idx[i,j], self.func_set[idx[i,j]].name, inpt)
                tmp[i,j] = self.func_set[idx[i,j]](*inpt)
        return tmp.sum()
    
    def update_parameters(self, reward, log_prob, entropy):# 更新参数
        R = torch.tensor(reward)                                # 倒序计算累计期望
        # loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        # print(log_probs[i], rewards[i], R, entropies[i])
        loss = -log_prob*R - 0.001*entropy

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)             # 梯度裁剪，梯度的最大L2范数=40
        self.optimizer.step()


ds = DeepSymbol(inpt_dim=4, hid_dim=32, func_set=func_set, lr=config.lr)
dir = './results/ckpt_deepsymbol_' + env_name

if not os.path.exists(dir):    
    os.mkdir(dir)

def rollout(env, policy:DeepSymbol, num_episode=config.rollout_episode):
    reward = 0
    for epi in range(num_episode):
        done = False
        state = env.reset()
        idx, log_prob, entropy = ds.sym_mat(torch.rand(1,4))
        for t in range(config.num_steps):
            action = policy.select_action(idx, state)
            state, r, done, _ = env.step(np.array([action]))
            reward += r
            if done: break
    return reward / num_episode, log_prob, entropy

for i_episode in range(config.num_episodes):
    reward, log_prob, entropy = rollout(env, ds)
    ds.update_parameters(reward, log_prob, entropy)
    if i_episode % config.ckpt_freq == 0:
        torch.save(ds.model.state_dict(), os.path.join(dir, 'VPG-'+str(i_episode)+'.pkl'))
    print("Episode: {}, reward: {}".format(i_episode, reward))

env.close()





