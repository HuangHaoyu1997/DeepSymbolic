'''
using CMA-ES to optimize symbol matrix

'''
import torch
import numpy as np
from core.function import func_set
import argparse, math, os, sys, gym, cma, ray, time, pickle
from copy import deepcopy

from core.utils import compute_centered_ranks, compute_weight_decay
from configuration import config
from env.CartPoleContinuous import CartPoleContinuousEnv
from core.DeepSymbol_v3 import DeepSymbol

env_name = 'CartPole-v1' # 'CartPoleContinuous'
# env = CartPoleContinuousEnv()

env_name = 'LunarLander-v2'
env = gym.make(env_name)
def wrapper(env, test):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    if not test:
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

inpt_dim = 6 # env.observation_space.shape[0]
out_dim = env.action_space.n

# env.seed(config.seed)
# torch.manual_seed(config.seed)
# np.random.seed(config.seed)
ray.init(num_cpus = config.num_parallel)

dir = './results/ckpt_deepsymbol-v31_' + env_name

if not os.path.exists(dir):    
    os.mkdir(dir)

@ray.remote
def rollout(env, ds:DeepSymbol, solution, num_episode=config.rollout_episode, test=False):
    policy = deepcopy(ds)
    policy.model.set_params(torch.tensor(solution[:policy.model.num_params]))
    policy.fc.set_params(solution[policy.model.num_params:])
    
    rewards, num_0 = [], []
    # sample N times from matrix distribution
    for _ in range(num_episode):
        idxs, _, _ = policy.sym_mat(test=False)
        zero_number = np.sum([(idx==7).sum().item() for idx in idxs])
        num_0.append(zero_number)
        # rollout N times for each sampling matrix
        for _ in range(num_episode):
            done = False
            # env = wrapper(env, test)
            # set seed for each episode for generality
            seed = int(str(time.time()).split('.')[1]) # if not test else config.seed
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            state = env.reset()
            rr = 0
            for _ in range(config.num_steps):
                action = policy.select_action(idxs, state[0:6])
                state, r, done, _ = env.step(action)
                # state, r, done, _ = env.step(np.array([action]))
                rr += r
                if done: break
            rewards.append(rr)
    
    if not test:
        return np.mean(rewards) - config.std_coef * np.std(rewards) + config.zero_coef * np.mean(num_0)
    else: # for test
        return np.mean(rewards), np.mean(num_0) # - np.std(rewards)

ds = DeepSymbol(inpt_dim, out_dim, func_set, num_mat=config.num_mat)
es = cma.CMAEvolutionStrategy([0.] * (ds.model.num_params + ds.fc.num_params),
                                config.sigma_init,
                                {'popsize': config.pop_size
                                    })

# training
for epi in range(config.num_episodes):
    tick = time.time()
    solutions = np.array(es.ask(), dtype=np.float32)
    rewards = [rollout.remote(env, ds, solution, config.rollout_episode, False) for solution in solutions]
    rewards = np.array(ray.get(rewards))
    ranks = compute_centered_ranks(rewards)
    # es.tell(solutions, -ranks)
    es.tell(solutions, -rewards)
    
    best_reward = rollout.remote(env, ds, es.result.xfavorite, config.rollout_episode, True)
    best_reward = ray.get(best_reward)
    
    print('episode:', epi, 'mean:', round(rewards.mean(), 2), round(rewards.std(), 2), 'max:', round(np.max(rewards),2), 'best:', *best_reward, 'time:', time.time()-tick)
    # print(rewards, ranks)
    if epi % config.ckpt_freq == 0:
        with open(os.path.join(dir, 'CMA_ES-'+str(epi)+'.pkl'), 'wb') as f:
            pickle.dump([es.result.xbest, es.result.xfavorite], f)

ray.shutdown()
