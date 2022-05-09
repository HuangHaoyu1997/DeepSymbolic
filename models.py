import torch
import torch.nn as nn
import torch.nn.functional as F

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