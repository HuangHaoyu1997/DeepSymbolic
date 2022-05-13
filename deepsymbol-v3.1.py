'''
using CMA-ES to optimize symbol matrix

'''
import torch
import numpy as np
from core.function import func_set
import argparse, math, os, sys, gym, cma, ray, time
from copy import deepcopy

from core.utils import compute_centered_ranks, compute_weight_decay
from configuration import config
from env.CartPoleContinuous import CartPoleContinuousEnv
from core.DeepSymbol_v3 import DeepSymbol

# env_name = 'CartPoleContinuous'
# env = CartPoleContinuousEnv()

env_name = 'LunarLander-v2'
env = gym.make(env_name)
inpt_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

env.seed(config.seed)
torch.manual_seed(config.seed)
np.random.seed(config.seed)
ray.init(num_cpus = config.num_parallel)

dir = './results/ckpt_deepsymbol-v3_' + env_name

if not os.path.exists(dir):    
    os.mkdir(dir)

@ray.remote
def rollout(env, ds, solution, num_episode=config.rollout_episode):
    policy = deepcopy(ds)
    policy.model.set_params(torch.tensor(solution))
    reward = 0
    for epi in range(num_episode):
        done = False
        state = env.reset()
        idx, log_prob, entropy = policy.sym_mat()
        for t in range(config.num_steps):
            action = policy.select_action(idx, state)
            state, r, done, _ = env.step(np.array([action]))
            reward += r
            if done: break
    return reward / num_episode

ds = DeepSymbol(inpt_dim, func_set)
es = cma.CMAEvolutionStrategy([0.] * ds.model.num_params,
                                config.sigma_init,
                                {'popsize': config.pop_size
                                    })

# training
for epi in range(config.num_episodes):
    tick = time.time()
    solutions = np.array(es.ask(), dtype=np.float32)
    results = []
    rewards = [rollout.remote(env, ds, solution) for solution in solutions]
    rewards = ray.get(rewards)
    rewards = np.array(rewards)

    best_policy_idx = np.argmax(rewards)
    best_policy = deepcopy(ds)
    best_policy.model.set_params(solutions[best_policy_idx])
    best_reward = rollout.remote(env, ds, solutions[best_policy_idx], 20)
    best_reward = ray.get(best_reward)

    ranks = compute_centered_ranks(rewards)
    es.tell(solutions, -ranks)
    print('episode:', epi, 'mean:', np.round(rewards.mean(), 2), 'max:', np.max(rewards), 'best:', best_reward, 'time:', time.time()-tick)
    # print(rewards, ranks)
    if epi % config.ckpt_freq == 0:
        torch.save(best_policy.model.state_dict(), os.path.join(dir, 'CMA_ES-'+str(epi)+'.pkl'))

ray.shutdown()
