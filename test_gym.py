import gym
import numpy as np
from core.utils import softmax
from env.CartPoleContinuous import CartPoleContinuousEnv
env = CartPoleContinuousEnv()
env = gym.make('CartPole-v1')

def policy_DSO(s):
    '''
        x2+exp(x3)
    ----------------------
    x2+exp(x3)+sin(x4)-1.1
    '''
    action = (s[1] + np.exp(s[2])) / (s[1] + np.exp(s[2]) + np.sin(s[3]) - 1.1)

    return action

def policy_SM(s):
    A = s[3] / s[2] + (s[2] + s[2] * np.sin(s[2])) / s[3]
    a1 = 3.7394824 * A - 11.637533
    a2 = 14.541853 * A - 0.35463876
    p = softmax([a1,a2])
    action = np.random.choice(2, p=p)
    
    return action

rrr = []
for _ in range(100):
    s = env.reset()
    done = False
    rr = 0
    while not done:
        # action = np.clip(policy_DSO(s), -1., 1.)
        action = policy_SM(s)
        s, r, done, info = env.step(action)
        rr += r
    print(rr)
    rrr.append(rr)
print(np.mean(rrr), np.std(rrr))

