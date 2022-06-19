import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

class Individual:
    def __init__(self, M, N, ) -> None:
        '''
        M: obs dim
        N: max variable number
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
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Update(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super(Update, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = layer_init(nn.Linear(2*hidden_dim, hidden_dim))

    def forward(self, aggr, hut_1):
        x = torch.cat((aggr, hut_1), -1)
        hut = F.relu(self.encoder(x))
        return hut

class GNN(nn.Module):
    def __init__(self, inpt_dim, hidden_dim, out_dim, Nnode) -> None:
        super(GNN, self).__init__()
        self.inpt_dim = inpt_dim
        self.out_dim = out_dim
        self.hid_dim = hidden_dim
        self.Nnode = Nnode
        
        self.encoding_fc1 = layer_init(nn.Linear(inpt_dim, hidden_dim))
        self.encoding_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.update_fc1 = Update(hidden_dim)
        self.update_fc2 = Update(hidden_dim)
        self.critic = layer_init(nn.Linear(hidden_dim, 1))
        self.actor_a = layer_init(nn.Linear(hidden_dim, out_dim))
        self.actor_b = layer_init(nn.Linear(hidden_dim, out_dim))
    
    def get_value(self, state, internal, graph):
        _, hu2 = self.ff(state, internal, graph)
        value = self.critic(hu2.sum(1))
        return value
    
    def get_action_and_value(self, state, internal, graph, action=None):
        hu1, hu2 = self.ff(state, internal, graph)
        value = self.critic(hu2.sum(1))
        alpha = F.softplus(self.actor_a(hu2.sum(1)))
        beta = F.softplus(self.actor_b(hu2.sum(1)))
        probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        
    def ff(self, state, internal, graph):
        batch_size = state.size(0)
        hu1 = torch.zeros((batch_size, self.Nnode, self.hid_dim))
        hu2 = torch.zeros((batch_size, self.Nnode, self.hid_dim))
        # layer 1
        # message sending
        state_embed = F.relu(self.encoding_fc1(state))
        internal_embed = F.relu(self.encoding_fc1(internal))
        
        # aggregation and update
        for key in graph:
            # aggregation
            state_neigh, internal_neigh = graph[key]
            aggregation = state_embed[:, state_neigh].sum(1) + internal_embed[:, internal_neigh].sum(1)
            # update
            hu1[:, key, :] = self.update_fc1(aggregation, internal_embed[:, key])
        # layer 2
        # message sending
        state_embed = F.relu(self.encoding_fc2(state_embed))
        internal_embed = F.relu(self.encoding_fc2(hu1))
        # aggregation and update
        for key in graph:
            # aggregation
            state_neigh, internal_neigh = graph[key]
            aggregation = state_embed[:, state_neigh].sum(1) + internal_embed[:, internal_neigh].sum(1)
            # update
            hu2[:, key, :] = self.update_fc2(aggregation, internal_embed[:, key])
        return hu1, hu2