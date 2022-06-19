import numpy as np
import random
from collections import deque

class Individual:
    def __init__(self, M, N, ) -> None:
        '''
        M: obs dim
        N: max varibale number
        '''
        self.M = M
        self.N = N
        self.L = (N+M)*N
        self.threshold = 0.5
        self.genetype = self.gene()
        self.fitness = 0.

    def gene(self,):
        '''generate gene'''
        tmp = np.zeros((self.L),dtype=np.int16)
        idx = np.array([random.random() for _ in range(self.L)])
        idx = np.where(idx>self.threshold)
        tmp[idx] = 1
        return tmp

def create_population(Npop, obs_dim, Nnode):
    '''generate population with size Npop'''
    return [Individual(obs_dim, Nnode) for _ in range(Npop)]

def translate(ind:Individual):
    '''map the individual genetype to a graph'''
    gene = ind.genetype
    Nnode, obs_dim = ind.N, ind.M
    adj_dict = {} # adjacent table
    for i in range(Nnode):
        gene_batch = gene[i*(Nnode + obs_dim):(i+1)*(Nnode + obs_dim)]
        gene_batch_1 = gene_batch[:obs_dim]
        gene_batch_2 = gene_batch[obs_dim:]
        obs_linked = np.where(gene_batch_1==1)[0] # state variables linked with this node
        var_linked = np.where(gene_batch_2==1)[0] # internal var linked with this node
        adj_dict[i] = [obs_linked, var_linked]
    return adj_dict

class StateVar:
    def __init__(self, maxlen) -> None:
        self.max_len = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.clear()
    def update(self, s):
        self.buffer.append(s)
    def clear(self,):
        self.buffer = deque(maxlen=self.max_len)
        [self.buffer.append(0.) for _ in range(self.max_len)]