import os, pickle, torch, random
import numpy as np
from core.trans import Encoder
import torch.optim as optim
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 120
d_obs = 13
d_embed = 16 # embedding dimension
n_heads = 8
d_k = 32
d_hidden = 32
d_class = 2
n_layers = 4
learning_rate = 1e-3
num_epoch = 40
encoder = Encoder(d_obs, d_embed, d_class, d_k, d_hidden, n_heads, n_layers).to(device)
optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

def data_generate(dir = './data/bad_trajectories.pkl', action_dim = 4):
    with open(dir, 'rb') as f:
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
# data_generate()
def get_batch_data(dir='./data/batch_data/'):
    train_data = []
    dlist = os.listdir(dir)
    for d in dlist:
        if d != 'test.pkl':
            with open(dir+d, 'rb') as f:
                train_data.extend(pickle.load(f))
    with open(dir+'test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, test_data

def test(epoch, test_data, model):
    test_acc = 0
    for i, x in enumerate(test_data):
        x = torch.FloatTensor(x).unsqueeze_(0).to(device)
        pred, _ = model(x)
        label = torch.tensor([1]) if i < N_test_samples else torch.tensor([0])
        if torch.argmax(pred.cpu()) != label:
            test_acc += 1
    print(epoch, 1 - test_acc/len(test_data))

train_data, test_data = get_batch_data()
N_train_samples = len(train_data)
false_data = np.random.rand(N_train_samples, 500, d_obs).tolist()
train_data.extend(false_data)

N_test_samples = len(test_data)
false_data = np.random.rand(N_test_samples, 500, d_obs).tolist()
test_data.extend(false_data)

idx = [i for i in range(2*N_train_samples)]
for epoch in range(num_epoch):
    random.shuffle(idx)
    loss, acc = 0, 0
    for i in range(2*N_train_samples):
        # index = idx[i*batch_size:(i+1)*batch_size]
        x = torch.FloatTensor(train_data[idx[i]]).unsqueeze_(0).to(device)
        pred, _ = encoder(x)
        label = torch.tensor([1]) if idx[i] < N_train_samples else torch.tensor([0])
        loss += F.nll_loss(pred, label.to(device))
        if torch.argmax(pred.cpu()) != label: # error counts
            acc += 1
        if (i+1) % batch_size == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, i, loss.cpu().item()/batch_size)
            loss = 0
        if i%1000 == 0:
            test(epoch, test_data, encoder)


