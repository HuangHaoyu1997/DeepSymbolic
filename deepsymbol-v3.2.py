'''
using CMA-ES to optimize symbol matrix
for Oil World Task

'''
import numpy as np
from core.function import func_set_OW
import os, sys, gym, cma, ray, time, pickle, torch, random
from copy import deepcopy

from core.utils import compute_centered_ranks, compute_weight_decay
from configuration import config
from core.DeepSymbol_v3 import DeepSymbol
from paves.scenarios.oil_world.oil_world import Oil_World
from paves.scenarios.oil_world.config import Oil_Config

run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19]

env_name = 'Oil' # 'CartPole-v1' # 'CartPoleContinuous'
logdir = './results/log-'+env_name+'-'+run_time+'.txt'


OW_config = Oil_Config()
env = Oil_World(OW_config, 1000, 5, 3, 3, history_len=5)
env.reset()
inpt_dim = len(env._get_modulate_obs())
out_dim = 1
ray.init(num_cpus = config.num_parallel)
dir = './results/ckpt_deepsymbol-v31_' + env_name

if not os.path.exists(dir):
    os.mkdir(dir)

@ray.remote
def rollout(env:Oil_World, ds:DeepSymbol, solution, num_episode=config.rollout_episode, test=False):
    def reward_func(env:Oil_World):
        if len(env.market.oil_price_his) <= 1:
            return 0
        if env.time % env.day_rounds != 0:
            return 0
        std = np.std(env.market.oil_price_his)
        if std < 20 and std > 150:
            return -1
        else:
            return 1
    policy = deepcopy(ds)
    policy.model.set_params(torch.tensor(solution[:policy.model.num_params]))
    policy.fc.set_params(solution[policy.model.num_params:])
    
    rewards, num_0 = [], []
    # sample N times from matrix distribution
    for _ in range(num_episode):
        idxs, _, _ = policy.sym_mat(test=False)
        zero_number = np.sum([(idx==8).sum().item() for idx in idxs])
        num_0.append(zero_number)
        # rollout N times for each sampling matrix
        for _ in range(num_episode):
            seed = int(str(time.time()).split('.')[1]) # if not test else config.seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            env.reset()
            rr = 0
            state = env._get_modulate_obs()
            for _ in range(config.num_steps):
                action = policy.select_action(idxs, state)
                env._set_modulate_action(action)
                rr += reward_func(env)
            rewards.append(rr)
    
    if not test:
        return np.mean(rewards) - config.std_coef * np.std(rewards) + config.zero_coef * np.mean(num_0)
    else: # for test
        return np.mean(rewards), np.mean(num_0) # - np.std(rewards)

ds = DeepSymbol(inpt_dim, out_dim, func_set_OW, num_mat=config.num_mat, is_discrete=False)
es = cma.CMAEvolutionStrategy([0.] * (ds.model.num_params + ds.fc.num_params),
                                config.sigma_init,
                                {'popsize': config.pop_size
                                    })

# training
for epi in range(config.num_episodes):
    if epi < 20:
        config.zero_coef = 0.05
    elif epi >= 20 and epi < 50:
        config.zero_coef = 0.4
    elif epi >= 50 and epi < 80:
        config.zero_coef = 0.6
    tick = time.time()
    solutions = np.array(es.ask(), dtype=np.float32)
    rewards = [rollout.remote(env, ds, solution, config.rollout_episode, False) for solution in solutions]
    rewards = np.array(ray.get(rewards))
    ranks = compute_centered_ranks(rewards)
    # es.tell(solutions, -ranks)
    es.tell(solutions, -rewards)
    
    best_reward = rollout.remote(env, ds, es.result.xfavorite, config.rollout_episode, True)
    best_reward = ray.get(best_reward)
    
    print('episode:', epi, 'mean:', round(rewards.mean(), 2), round(rewards.std(), 2), \
        'max:', round(np.max(rewards),2), 'best:', *best_reward, 'time:', time.time()-tick)
    # print(rewards, ranks)
    

    with open(logdir,'a+') as f:
        f.write(
            str(epi)+' mean:'+str(round(rewards.mean(), 2))+' '+str(round(rewards.std(), 2))+\
            ' max:'+str(round(np.max(rewards),2))+' best:'+str(best_reward[0])+' '+str(best_reward[1])+\
            ' time:'+str(time.time()-tick)+'\n'
            )

    if epi % config.ckpt_freq == 0:
        with open(os.path.join(dir, 'CMA_ES-'+str(epi)+'.pkl'), 'wb') as f:
            pickle.dump([es.result.xbest, es.result.xfavorite], f)

ray.shutdown()
