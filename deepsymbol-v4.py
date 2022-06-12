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
    def gene(self,):
        '''generate gene'''
        tmp = np.zeros((self.L),dtype=np.int16)
        idx = [random.random() for _ in range(self.L)]
        return