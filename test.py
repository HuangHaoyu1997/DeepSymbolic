import os, pickle, torch, random
import numpy as np
from core.trans import Encoder
import torch.optim as optim
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 40
d_obs = 13
d_embed = 16 # embedding dimension
n_heads = 8
d_k = 32
d_hidden = 32
d_class = 2
n_layers = 4
learning_rate = 1e-3
num_epoch = 20
encoder = Encoder(d_obs, d_embed, d_class, d_k, d_hidden, n_heads, n_layers).to(device)
optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

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
false_data = np.random.rand(num_samples, 500, d_obs).tolist()
data.extend(false_data)

idx = [i for i in range(2*num_samples)]
for epoch in range(num_epoch):
    random.shuffle(idx)
    loss = 0
    for i in range(2*num_samples):
        # index = idx[i*batch_size:(i+1)*batch_size]
        x = torch.FloatTensor(data[i]).unsqueeze_(0)
        pred, _ = encoder(x)
        label = torch.tensor([1]) if i<num_samples else torch.tensor([0])
        loss += F.nll_loss(pred, label)
        if (i+1) % batch_size == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
            loss = 0

