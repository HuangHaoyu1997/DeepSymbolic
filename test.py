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
    true_data = []
    dlist = os.listdir(dir)
    for d in dlist:
        if d != 'test.pkl':
            with open(dir+d, 'rb') as f:
                true_data.extend(pickle.load(f))
    Ntrue = len(true_data)
    with open(dir+'test.pkl', 'rb') as f:
        false_data = pickle.load(f)
    true_data.extend(false_data)

    label = np.ones((len(true_data)), dtype=np.int16)
    label[Ntrue:] = 0
    return true_data, label

def test(epoch, test_data, test_label, model:Encoder):
    model.eval()
    test_err = 0
    for x, y in zip(test_data, test_label):
        x = torch.FloatTensor(x).unsqueeze_(0).to(device)
        pred, _ = model(x)
        label = torch.tensor(y)
        if torch.argmax(pred.cpu()) != label:
            test_err += 1
    print(epoch, 1 - test_err/len(test_data))

data, label = get_batch_data()


def shuffle(data, label):
    N_samples = len(data)
    idx = [i for i in range(N_samples)]
    random.shuffle(idx)
    train_data = [data[i] for i in idx[:int(0.8*N_samples)]]
    train_label = [label[i] for i in idx[:int(0.8*N_samples)]]
    test_data = [data[i] for i in idx[int(0.8*N_samples):]]
    test_label = [label[i] for i in idx[int(0.8*N_samples):]]
    return train_data, train_label, test_data, test_label

train_data, train_label, test_data, test_label = shuffle(data, label)
Ntrain = len(train_data)
Ntest = len(test_data)
for epoch in range(num_epoch):
    idx = [i for i in range(Ntrain)]
    random.shuffle(idx)
    loss, acc = 0, 0
    for i in range(Ntrain):
        
        x = torch.FloatTensor(train_data[idx[i]]).unsqueeze_(0).to(device)
        pred, _ = encoder(x)
        label = torch.LongTensor([train_label[idx[i]]])
        loss += F.nll_loss(pred, label.to(device))
        if torch.argmax(pred.cpu()) != label: # error counts
            acc += 1
        if (i+1) % batch_size == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, i, loss.cpu().item()/batch_size)
            loss = 0
        if i % 100 == 0:
            test(epoch, test_data, test_label, encoder)


