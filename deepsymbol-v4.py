from copy import deepcopy
import gym, pickle, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from algorithms.utils_v4 import Individual, create_population, translate, StateVar


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
    def __init__(self, inpt_dim, hidden_dim, out_dim) -> None:
        super(GNN, self).__init__()
        self.inpt_dim = inpt_dim
        self.out_dim = out_dim
        self.hid_dim = hidden_dim
        
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
        hu1 = torch.zeros((batch_size, Nnode, self.hid_dim))
        hu2 = torch.zeros((batch_size, Nnode, self.hid_dim))
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





import os, random, time, ray
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
import warnings
warnings.filterwarnings('ignore')

class Args:
    verbose = False
    exp_name = os.path.basename(__file__).rstrip(".py")
    seed = 123
    num_cpus = 6
    torch_deterministic = True
    cuda = False
    env_id = "LunarLanderContinuous-v2" # 'BipedalWalker-v3'
    total_timesteps = 500000
    learning_rate = 3e-4
    num_envs = 8 # the number of parallel game environments
    num_steps = 300 # the number of steps to run in each environment per policy rollout
    anneal_lr = True
    gae = True
    gae_lambda = 0.95
    gamma = 0.99
    num_minibatches = 32
    update_epochs = 10 # K epochs to update the policy
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True 
    ent_coef = 0.0 # coefficient of the entropy
    vf_coef = 0.5 # coefficient of the value function
    max_grad_norm = 0.5 # the maximum norm for the gradient clipping
    target_kl = None # the target KL divergence threshold
    batch_size = int(num_envs * num_steps) # 8*2048
    minibatch_size = int(batch_size // num_minibatches)

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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_A = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)
        self.actor_B = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)
        
    def get_value(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.critic(x)
        return x

    def get_action_and_value(self, x, action=None):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.critic(x)
        alpha = F.softplus(self.actor_A(x))
        beta = F.softplus(self.actor_B(x))
        probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

def set_seed(args:Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

# @ray.remote
def main(args:Args):
    warnings.filterwarnings('ignore')
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    set_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # start the game
    global_step = 0
    start_time = start_time_ = time.time()
    
    next_obs = torch.Tensor(envs.reset(seed=args.seed)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size # total update counts

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy()*2-1)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            if 'episode' in info.keys(): 
                for item in info['episode']:
                    if item is not None:
                        current_r = item['r']
                        if args.verbose:
                            print(f"global_step={global_step}, episodic_return={item['r']}, time={time.time()-start_time}")
                            start_time = time.time()
                        break
            # for item in info:
            #     # if "episode" in item.keys():
            #     if item == "episode":
            #         print(info[item])
            #         current_r = info[item]['r']
            #         if args.verbose: 
            #             print(f"global_step={global_step}, episodic_return={item['episode']['r']}, time={time.time()-start_time}")
            #             start_time = time.time()
            #         break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        if args.verbose: print('start training')
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl: break
    envs.close()
    print('training time:', time.time() - start_time_)
    return current_r, time.time() - start_time_


if __name__ == '__main__':
    
    
    generation = 10
    args = Args()
    # ray.init(num_cpus=args.num_cpus)
    # for _ in range(generation):
    #     run_id = [main.remote(args) for _ in range(5)]
    #     reward = ray.get(run_id)
    #     print(reward)
    # ray.shutdown()
    for _ in range(5):
        current_r, time_consuming = main(args)
        print(current_r, time_consuming)
    
    
    
    
    
    
    
    
    
    
    
    
    # pop = create_population(10, 5, 13)
    # adj_dict = translate(pop[0])
    # for i in adj_dict:
    #     print(adj_dict[i])
    
    Nnode = 5
    max_len = 1
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
    # print(graphs[0])

    Internal_var = torch.rand((2, Nnode, max_len), dtype=torch.float32)
    model = GNN(inpt_dim=max_len, hidden_dim=hid_dim, out_dim=5)
    state = torch.tensor([[svar.buffer for svar in state_vars],
                          [svar.buffer for svar in state_vars]])
    
    hus1, hus2, out = model(state, Internal_var, graphs[0])
    print(out)


