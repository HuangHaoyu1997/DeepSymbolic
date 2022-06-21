import numpy as np
import random, gym
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from copy import deepcopy

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

class GNN(nn.Module):
    def __init__(self, inpt_dim, hidden_dim, out_dim, Nnode) -> None:
        super(GNN, self).__init__()
        self.inpt_dim = inpt_dim
        self.out_dim = out_dim
        self.hid_dim = hidden_dim
        self.Nnode = Nnode
        
        self.encoding_fc1 = layer_init(nn.Linear(inpt_dim, hidden_dim))
        self.encoding_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.encoding_fc3 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.update_fc1 = Update(hidden_dim)
        self.update_fc2 = Update(hidden_dim)
        self.update_fc3 = Update(hidden_dim)
        self.critic = layer_init(nn.Linear(hidden_dim, 1))
        self.actor_a = layer_init(nn.Linear(hidden_dim, out_dim))
        self.actor_b = layer_init(nn.Linear(hidden_dim, out_dim))
    
    def get_value(self, state, internal, graph):
        hu = self.ff(state, internal, graph)
        value = self.critic(hu[-1].sum(1))
        return value
    
    def get_action_and_value(self, state, internal, graph, action=None):
        hu = self.ff(state, internal, graph)
        value = self.critic(hu[-1].sum(1))
        alpha = F.softplus(self.actor_a(hu[-1].sum(1)))
        beta = F.softplus(self.actor_b(hu[-1].sum(1)))
        alpha += 1e-3
        beta += 1e-3
        # print(alpha.shape, beta.shape)
        probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        
    def ff(self, state, internal, graph):
        batch_size = state.size(0)
        hu1 = torch.zeros((batch_size, self.Nnode, self.hid_dim))
        hu2 = torch.zeros((batch_size, self.Nnode, self.hid_dim))
        hu3 = torch.zeros((batch_size, self.Nnode, self.hid_dim))
        # layer 1
        state_embed = F.relu(self.encoding_fc1(state)) # message sending
        internal_embed = F.relu(self.encoding_fc1(internal))
        for key in graph: # aggregation and update
            state_neigh, internal_neigh = graph[key]
            aggregation = state_embed[:, state_neigh].sum(1) + internal_embed[:, internal_neigh].sum(1)
            hu1[:, key, :] = self.update_fc1(aggregation, internal_embed[:, key]) # update
        # layer 2
        state_embed = F.relu(self.encoding_fc2(state_embed))
        internal_embed = F.relu(self.encoding_fc2(hu1))
        for key in graph:
            state_neigh, internal_neigh = graph[key]
            aggregation = state_embed[:, state_neigh].sum(1) + internal_embed[:, internal_neigh].sum(1)
            hu2[:, key, :] = self.update_fc2(aggregation, internal_embed[:, key])
        # layer3
        state_embed = F.relu(self.encoding_fc3(state_embed))
        internal_embed = F.relu(self.encoding_fc3(hu2))
        for key in graph:
            state_neigh, internal_neigh = graph[key]
            aggregation = state_embed[:, state_neigh].sum(1) + internal_embed[:, internal_neigh].sum(1)
            hu3[:, key, :] = self.update_fc3(aggregation, internal_embed[:, key])
        
        
        return hu1, hu2, hu3

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


def make_env_sac(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk
def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


