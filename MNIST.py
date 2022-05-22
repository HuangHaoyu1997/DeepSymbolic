'''
pytorch tutorial
created by HHY,2021年3月30日01:49:44
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

n_epochs = 10
batch_size_train = 3    # batch size when training
batch_size_test = 1000   # batch size when testing
learning_rate = 0.01
momentum = 0.5           # hyper-parameter of SGD
log_interval = 10        # print training log every 10 iterations
random_seed = 1          # fixed seed for experiment reproducibility
torch.manual_seed(random_seed)

# You can change the dataset by replace "torchvision.datasets.MNIST" with "torchvision.datasets.CIFAR10" or "torchvision.datasets.FashionMNIST".
# I have already downloaded these 3 datasets in ./data/ 
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, # ./data/  is directory of dataset，train=True for training dataset
                                                                        transform=torchvision.transforms.Compose([     
                                                                        torchvision.transforms.ToTensor(),  # data type transformation from numpy to torch.tensor
                                                                        torchvision.transforms.Normalize(   # x = (x - mean(x))/stddev(x)
                                                                        (0.1307,), (0.3081,))               # 0.1307 for mean，0.3081 for stddev
                                                                        ])),
                                            batch_size=batch_size_train, shuffle=True)                      # before starting every epoch，shuffle the training set

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                                                                        transform=torchvision.transforms.Compose([
                                                                        torchvision.transforms.ToTensor(),
                                                                        torchvision.transforms.Normalize(
                                                                        (0.1307,), (0.3081,))
                                                                        ])),
                                            batch_size=batch_size_test, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')

# train(1) # train only 1 epoch

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

