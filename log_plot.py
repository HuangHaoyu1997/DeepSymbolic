import matplotlib.pyplot as plt
import numpy as np

with open('C:/Users/44670/Documents/GitHub/DeepSymbolic/results/log-LunarLander-v2-2022-05-20-02-05-51.txt', 'r+') as f:
    log = f.readlines()

test_reward = []
moving_avg = []
for i, l in enumerate(log):
    score = float(l.split('best:')[1].split(' ')[0])
    if i==0: moving_avg.append(score)
    else: moving_avg.append(moving_avg[-1]*0.95 + score*0.05)
    test_reward.append(score)
plt.plot(test_reward)
plt.plot(moving_avg)
plt.xlabel('Episode'); plt.ylabel('Reward')
plt.grid()
plt.show()
