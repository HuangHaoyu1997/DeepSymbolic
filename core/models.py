import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, inpt_dim, dict_dim, ):
        super(Model, self).__init__()
        self.inpt_dim = inpt_dim 
        self.dict_dim = dict_dim
        self.fc1 = nn.Linear(1, inpt_dim*inpt_dim*self.dict_dim)
        self.fc2 = nn.Linear(1, inpt_dim*inpt_dim*self.dict_dim)
        self.fc3 = nn.Linear(1, inpt_dim*inpt_dim*self.dict_dim)

    def forward(self, ):
        mat1 = self.fc1(torch.tensor([0.]))
        mat1 = mat1.view(self.inpt_dim, self.inpt_dim, self.dict_dim)
        mat1 = F.softmax(mat1, dim=-1) # mat1.shape=(4,4,5)

        mat2 = self.fc2(torch.tensor([0.]))
        mat2 = mat2.view(self.inpt_dim, self.inpt_dim, self.dict_dim)
        mat2 = F.softmax(mat2, dim=-1) # mat2.shape=(4,4,5)

        mat3 = self.fc3(torch.tensor([0.]))
        mat3 = mat3.view(self.inpt_dim, self.inpt_dim, self.dict_dim)
        mat3 = F.softmax(mat3, dim=-1) # mat3.shape=(4,4,5)
        return mat1, mat2, mat3
    
    @property
    def num_params(self):
        return self.fc1.bias.data.size()[0]+self.fc2.bias.data.size()[0]+self.fc3.bias.data.size()[0]
        # return sum([np.prod(params.size()) for params in self.state_dict().values()])
    
    def get_params(self):
        
        return torch.cat([self.fc1.bias.data, self.fc2.bias.data, self.fc3.bias.data])
        # return torch.cat([params.flatten() for params in self.state_dict().values()])
    
    def set_params(self, all_params):
        '''
        给各fc层的bias写入参数
        各fc层的weight写入0
        '''
        all_params = torch.FloatTensor(all_params)
        state_dict = dict()
        for key, params in self.state_dict().items():
            if key.split('.')[1]=='weight':
                state_dict[key] = torch.zeros_like(params)
            if key.split('.')[1]=='bias':
                i = int(key.split('.')[0][-1])
                state_dict[key] = all_params[(i-1)*len(params):i*len(params)]
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    
    model = Model(2, 3)
    print(model.set_params(torch.rand(36)))
    print(model.fc3.weight.data)