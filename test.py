import os, pickle
import numpy as np
from core.trans import Encoder

batch_size = 16
d_obs = 13
d_embed = 16 # embedding dimension
n_heads = 8
d_k = 32
d_hidden = 32
d_class = 2
n_layers = 4
encoder = Encoder(d_obs, d_embed, d_class, d_k, d_hidden, n_heads, n_layers)

def data_generate(dir = './data/', action_dim = 4):
    with open(dir+'good_trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    batch_data = []
    for i, traj in enumerate(trajectories):
        new_traj = []
        for transition in traj:
            one_hot = np.zeros((action_dim), dtype=np.float32)
            action = transition[1]
            one_hot[action] = 1
            tran = np.concatenate((transition[0], one_hot, [transition[2]]))
            new_traj.append(tran)
        batch_data.append(new_traj)
        if len(batch_data)==1000:
            with open('./data/batch_'+str(i+1)+'.pkl', 'wb') as f:
                pickle.dump(batch_data, f)
                batch_data = []
    with open('./data/batch_data/batch_'+str(i+1)+'.pkl', 'wb') as f:
        pickle.dump(batch_data, f)

def get_batch_data(dir='./data/batch_data/'):
    total_data = []
    dlist = os.listdir(dir)
    for d in dlist:
        with open(dir+d, 'rb') as f:
            total_data.extend(pickle.load(f))
    return total_data
data = get_batch_data()
num_samples = len(data)
iterations = num_samples // batch_size
fake_data = np.random.rand(5000, 500, 13)

for i in range(iterations):
    

