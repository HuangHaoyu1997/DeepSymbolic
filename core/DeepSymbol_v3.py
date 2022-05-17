'''
根据Model输出的symbol operator matrix
以及environment给出的observation
计算出相应的action
'''

import torch
import numpy as np
from torch.distributions import Categorical
# from .utils import tanh
# import sys
# sys.path.append(r'C:/Users/44670/Documents/GitHub/DeepSymbolic')
from core.models import Model, Linear

class DeepSymbol():
    def __init__(self, inpt_dim, out_dim, func_set) -> None:
        self.inpt_dim = inpt_dim
        self.out_dim = out_dim
        self.func_set = func_set
        self.dict_dim = len(func_set)
        self.model = Model(inpt_dim = self.inpt_dim, dict_dim= self.dict_dim)
        self.fc = Linear(inpt_dim, out_dim)
        # self.model.train()
    
    def select_action(self, idxs, state):
        # state = torch.tensor(state)
        _, action1 = self.execute_symbol_mat(state, idxs)
        action2 = self.fc(action1[2].numpy())
        action3 = np.random.choice(np.arange(self.out_dim), p=action2)
        print(action1, action2, action3, '\n')
        # action = tanh(action, alpha=0.05)
        # print(action,'\n')
        return action3

    def sym_mat(self, test=False):
        '''get symbol matrix for policy'''
        mats = self.model()
        if not test:
            dist = [Categorical(mat) for mat in mats]
            idxs = [p.sample() for p in dist]
            log_prob = torch.sum(torch.tensor([p.log_prob(idx).sum() for p, idx in zip(dist, idxs)]))
            entropies = torch.sum(torch.tensor([p.entropy().log().sum() for p in dist]))
        elif test:
            mat1, mat2, mat3 = self.model()
            _, mat1_idx = torch.max(mat1, dim=-1)
            _, mat2_idx = torch.max(mat2, dim=-1)
            _, mat3_idx = torch.max(mat3, dim=-1)
            idxs = [mat1_idx, mat2_idx, mat3_idx]
            log_prob, entropies = None, None
        # mat1, mat2, mat3 = self.model()
        # p1 = Categorical(mat1)
        # p2 = Categorical(mat2)
        # p3 = Categorical(mat3)
        # idx1 = p1.sample()
        # idx2 = p2.sample()
        # idx3 = p3.sample()
        # idxs = [idx1, idx2, idx3]
        # log_prob = p1.log_prob(idx1).sum() + p2.log_prob(idx2).sum() + p3.log_prob(idx3).sum()
        # entropies = p1.entropy().log().sum() + p2.entropy().log().sum() + p3.entropy().log().sum()
        # print(log_prob, entropies)
        
        return idxs, log_prob, entropies
    
    def execute_symbol_mat(self, state, idxs):
        '''symbolic calculation using state vector'''
        state = torch.tensor(state)
        internal_output = torch.zeros((len(idxs), 
                                  self.inpt_dim, 
                                  self.inpt_dim), dtype=torch.float32)
        for ii, idx in enumerate(idxs):
            internal_input = state if ii==0 else internal_output[ii-1].sum(0)
            print(internal_input)

            for i in range(self.inpt_dim):
                for j in range(self.inpt_dim):
                    arity = self.func_set[idx[i,j]].arity
                    
                    if arity == 1: inpt = [internal_input[i]]
                    elif arity == 2: inpt = [internal_input[i], internal_input[j]]
                    
                    internal_output[ii,i,j] = self.func_set[idx[i,j]](*inpt)
            
        return internal_output.sum(1)

if __name__ == '__main__':
    from core.function import func_set
    ds = DeepSymbol(4, func_set)
    idxs, _, _ = ds.sym_mat()
    print(ds.execute_symbol_mat([1., 2., 3., 4.], idxs))

    from core.DeepSymbol_v3 import DeepSymbol
    from core.function import func_set
    ds = DeepSymbol(4, 4, func_set)
    idxs, _, _ = ds.sym_mat()
    print(ds.select_action(idxs, [1.,2.,3.,4.]))
    print(ds.fc.num_params)

