import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    '''
    Model for version 3.0
    '''
    def __init__(self, inpt_dim, dict_dim, num_mat):
        super(Model, self).__init__()
        self.inpt_dim = inpt_dim 
        self.dict_dim = dict_dim
        self.num_mat = num_mat
        self.fcs = [nn.Linear(1, inpt_dim*inpt_dim*self.dict_dim)] * num_mat

    def forward(self, ):
        x = torch.tensor([0.])
        mats = []
        for fc in self.fcs:
            mat = fc(x)
            mat = mat.view(self.inpt_dim, self.inpt_dim, self.dict_dim)
            mat = F.softmax(mat, dim=-1)
            mats.append(mat)
        
        return mats
    
    @property
    def num_params(self):
        # 鉴于输入fc的是常数0, 有用参数仅为各fc的bias
        count = 0
        for fc in self.fcs:
            count += fc.bias.data.size()[0]
        return count
        # return sum([np.prod(params.size()) for params in self.state_dict().values()])
    
    def get_params(self):
        return torch.cat([fc.bias.data for fc in self.fcs])
        # return torch.cat([params.flatten() for params in self.state_dict().values()])
    
    def set_params(self, all_params):
        '''
        给各fc层的bias写入参数
        各fc层的weight写入0
        '''
        all_params = torch.tensor(all_params, dtype=torch.float32)
        length = len(all_params)//self.num_mat
        for i, fc in enumerate(self.fcs):
            fc.weight.data = torch.zeros(fc.weight.data.shape)
            fc.bias.data = all_params[i*length:(i+1)*length]


class Linear:
    '''Linear layer implementation using numpy'''
    def __init__(self, in_dim, out_dim) -> None:
        self.weight = np.zeros((in_dim, out_dim), dtype=np.float32)
        self.bias = np.zeros((out_dim), dtype=np.float32)
        
    def relu(self, x):
        return np.maximum(x, 0)
    
    def softmax(self, x, alpha=1.0, with_clip=50):
        x = np.clip(x, -with_clip, with_clip)
        return np.exp(alpha*x)/(np.exp(alpha*x)).sum()

    def __call__(self, x):
        Y = np.matmul(x, self.weight) + self.bias
        return self.softmax(self.relu(Y))

    @property
    def num_params(self,):
        return self.weight.size + self.bias.size

    def set_params(self, param):
        weight_param = param[:self.weight.size]
        bias_param = param[self.weight.size:]
        self.weight = weight_param.reshape(self.weight.shape)
        self.bias = bias_param.reshape(self.bias.shape)

if __name__ == '__main__':
    
    model = Model(2, 3, 4)
    
    # print(model.get_params())
    model.set_params(torch.ones(48))
    print(model.fcs[0].weight.data, model.fcs[0].bias.data)
    print(model())
    # print(model.fc3.weight.data)