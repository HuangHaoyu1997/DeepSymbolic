from collections import deque
from copy import deepcopy
import gym, pickle, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class GAT(nn.Module):
    def __init__(self, inpt_dim, hidden_dim, out_dim) -> None:
        super(GAT, self).__init__()
        self.inpt_dim = inpt_dim
        self.out_dim = out_dim
        self.hid_dim = hidden_dim
        
        self.encoding_fc = nn.Linear(inpt_dim, hidden_dim)
        self.update_fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, state_vars, internal, graph):
        state = torch.tensor([svar.buffer for svar in state_vars])
        
        state_embed = F.relu(self.encoding_fc(state))
        internal_embed = F.relu(self.encoding_fc(internal))
        
        for key in graph:
            state_neigh, var_neigh = graph[key]
            print(state_neigh, var_neigh)
            print(state_embed[state_neigh].sum(0))
        return state_embed, internal_embed


class ES:
    def __init__(self,
                pop_size,
                mutation_rate,
                crossover_rate,
                obs_dim,
                Nnode,
                elite_rate,
                ) -> None:
        self.pop_size = pop_size
        self.mut_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.pop = create_population(pop_size, obs_dim, Nnode)
        self.elite_rate = elite_rate
        self.elite_pop = int(pop_size * elite_rate)
    
    def ask(self,):
        # solutions = [ind.genetype for ind in self.pop]
        # return solutions
        return self.pop

    def crossover(self, ind1:Individual, ind2:Individual):
        idx = np.array([random.random() for _ in range(ind1.L)])
        idx = np.where(idx<self.cross_rate)
        cross_batch1 = deepcopy(ind1.genetype[idx])
        cross_batch2 = deepcopy(ind2.genetype[idx])
        ind1.genetype[idx] = cross_batch2
        ind2.genetype[idx] = cross_batch1

    def mutation(self, ind:Individual):
        ind1 = deepcopy(ind)
        idx = np.array([random.random() for _ in range(ind1.L)])
        idx = np.where(idx < self.mut_rate)
        ind1.genetype[idx] = 1 - ind1.genetype[idx]
        return ind1

    def tell(self, fitness):
        for ind, fit in zip(self.pop, fitness):
            ind.fitness = fit
        new_pop = sorted(self.pop, key=lambda ind: ind.fitness)[::-1]
        elite_pop = new_pop[:self.elite_pop]
        child_pop = []
        for _ in range(self.pop_size-self.elite_pop):
            parent = random.choice(elite_pop)
            child_pop.append(self.mutation(parent))
        elite_pop.extend(child_pop)
        self.pop = elite_pop
        # for ind in self.pop:
        #     ind.fitness = 0.

if __name__ == '__main__':
    # pop = create_population(10, 5, 13)
    # adj_dict = translate(pop[0])
    # for i in adj_dict:
    #     print(adj_dict[i])
    
    Nnode = 10
    max_len = 5
    hid_dim = 6
    env = gym.make('BipedalWalker-v3')
    state = env.reset()
    state_vars = [StateVar(max_len) for _ in state]
    [svar.update(s) for svar, s in zip(state_vars, state)]
    
    es = ES(pop_size=100,
            mutation_rate=0.5,
            crossover_rate=0.5,
            obs_dim=2,
            Nnode=Nnode,
            elite_rate=0.15)
    
    pop = es.ask()
    # es.tell(np.random.rand(100))
    graphs = [translate(ind) for ind in pop]

    Internal_var = torch.zeros((Nnode, max_len), dtype=torch.float32)
    model = GAT(inpt_dim=max_len, hidden_dim=hid_dim, out_dim=5)
    s,i = model(state_vars, Internal_var, graphs[0])
    # print(s,i)

    

