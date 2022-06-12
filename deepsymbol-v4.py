import gym, pickle, random
import numpy as np

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

    def gene(self,):
        '''generate gene'''
        tmp = np.zeros((self.L),dtype=np.int16)
        idx = np.array([random.random() for _ in range(self.L)])
        idx = np.where(idx>self.threshold)
        tmp[idx] = 1
        return tmp

def population(Npop, M, N):
    '''generate population with size Npop'''
    return [Individual(M,N) for _ in range(Npop)]

def translate(ind:Individual):
    '''map the individual genetype to a graph'''
    gene = ind.genetype
    N, M = ind.N, ind.M
    adj_dict = {}
    for i in range(N):
        gene_batch = gene[i*(N+M):(i+1)*(N+M)]
        