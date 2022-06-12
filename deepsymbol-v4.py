from collections import deque
from turtle import forward
import gym, pickle, random
import numpy as np
import torch
import torch.nn as nn

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

def population(Npop, obs_dim, Nnode):
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
        [self.buffer.append(00.) for _ in range(self.max_len)]

class GAT(nn.Module):
    def __init__(self) -> None:
        super(GAT, self).__init__()
        self.fc = nn.Linear(10,2)
    def forward(self, x):
        return self.fc(x)

class SimpleGA:
    '''Simple Genetic Algorithm.'''
    def __init__(self, num_params,      # number of model parameters
                sigma_init=0.1,        # initial standard deviation
                sigma_decay=0.999,     # anneal standard deviation
                sigma_limit=0.01,      # stop annealing if less than this
                popsize=256,           # population size
                elite_ratio=0.1,       # percentage of the elites
                forget_best=False,     # forget the historical best elites
                weight_decay=0.01,     # weight decay coefficient
                ):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.popsize = popsize

        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

        self.sigma = self.sigma_init
        self.elite_params = np.zeros((self.elite_popsize, self.num_params))
        self.elite_rewards = np.zeros(self.elite_popsize)
        self.best_param = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay

    def rms_stdev(self):
        return self.sigma # same sigma for all parameters.

    def ask(self):
        '''returns a list of parameters'''
        self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
        solutions = []
    
        def mate(a, b):
            c = np.copy(a)
            idx = np.where(np.random.rand((c.size)) > 0.5)
            c[idx] = b[idx]
            return c
    
        elite_range = range(self.elite_popsize)
        for i in range(self.popsize):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = mate(self.elite_params[idx_a], self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)
    
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        if self.forget_best or self.first_iteration:
            reward = reward_table
            solution = self.solutions
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]
    
        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.best_param

    def result(self): # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)

class ES:
    def __init__(self,
                pop_size,
                mutation_rate,
                obs_dim,
                Nnode,
                elite_rate,
                ) -> None:
        self.pop_size = pop_size
        self.mut_rate = mutation_rate
        self.pop = population(pop_size, obs_dim, Nnode)
        self.elite_rate = elite_rate
        self.elite_pop = pop_size * elite_rate
    
    def ask(self,):
        return self.pop

    def mutation(self,):
        pass

    def tell(self, fitness):
        
if __name__ == '__main__':
    # pop = population(10, 5, 13)
    # adj_dict = translate(pop[0])
    # for i in adj_dict:
    #     print(adj_dict[i])
    
    # env = gym.make('BipedalWalker-v3')
    # state = env.reset()
    # state_var_list = [StateVar(maxlen=10) for _ in state]
    # [svar.update(s) for svar,s in zip(state_var_list, state)]
    # done = False
    # while not done:
    #     state, r, done, _ = env.step(env.action_space.sample())
    model = GAT()
    print(model(torch.ones(1,10)))
