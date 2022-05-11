import torch 
from torch.distributions import Categorical
from utils import tanh
from models import Model

class DeepSymbol():
    def __init__(self, inpt_dim, func_set) -> None:
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
