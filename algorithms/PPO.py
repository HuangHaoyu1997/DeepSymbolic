import torch, pickle
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Softmax()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_prob = self.actor(state)
        dist = Categorical(probs=action_prob)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_prob = torch.squeeze(self.actor(state))
        dist = Categorical(probs=action_prob)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
class PPO:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().item()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.FloatTensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = (ratios * advantages).float()
            surr2 = (torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages).float()
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            '''
            print(torch.min(surr1, surr2),
            type(torch.min(surr1, surr2)),
            type(self.MseLoss(state_values, rewards)),
            type(dist_entropy))
            '''
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2" # BipedalWalker-v3
    render = False
    solved_reward = 500         # stop training if avg_reward > solved_reward
    log_interval = 1           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 500        # max timesteps in one episode
    
    update_timestep = 2000      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 3e-4                   # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = 234
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if random_seed:
        # print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip)
    
    # logging variables
    time_step = 0
    
    # training loop
    data_storage = []
    for i_episode in range(1, max_episodes+1):
        env.seed(random_seed+i_episode)
        state = env.reset()
        trajectory = []
        reward = 0
        for t in range(max_timesteps):
            time_step += 1
            action = ppo.select_action(state, memory)
            next_state, r, done, _ = env.step(action)
            trajectory.append([state, action, r])
            state = next_state
            # Saving reward and is_terminals:
            memory.rewards.append(r)
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            reward += r
            if render: env.render()
            if done: break
        if reward < 50.0:
            data_storage.append(trajectory)
            print(len(data_storage), reward)
        if len(data_storage) % 20 == 0:
            with open('./data/bad_trajectories.pkl', 'wb') as f:
                pickle.dump(data_storage, f)

        # stop training if avg_reward > solved_reward
        if reward > solved_reward:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './results/PPO_continuous_{}.pth'.format(env_name))
            
        # logging
        reward = round(reward, 2)
        print('Episode {} \t Epi length: {} \t Epi reward: {}'.format(i_episode, t, reward))
        
        
if __name__ == '__main__':
    main()