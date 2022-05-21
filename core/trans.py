import numpy as np
import torch
import torch.nn as nn
len_traj = 13
batch_size = 3
d_obs = 6
d_embed = 7 # embedding dimension
n_heads = 8
d_k = 16
d_hidden = 16
d_classification = 2
n_layers = 4 # Encoder内含
trajectory = torch.rand(batch_size, len_traj, d_obs)

class Embedding(nn.Module):
    '''将轨迹序列映射到隐空间'''
    def __init__(self, inpt_dim, embed_dim):
        super(Embedding, self).__init__()
        self.fc = nn.Linear(inpt_dim, embed_dim, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V):
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)] [1,8,5,5]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embed, d_k * n_heads) # d_embed,7维, d_k,16*8=128维
        self.W_K = nn.Linear(d_embed, d_k * n_heads)
        self.W_V = nn.Linear(d_embed, d_k * n_heads)
        self.fc = nn.Linear(n_heads * d_k, d_embed)
        self.layer_norm = nn.LayerNorm(d_embed)

    def forward(self, x):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = x, x.size(0) # 残差跨层连接
        
        q_s = self.W_Q(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_q x d_k]
        v_s = self.W_V(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_q x d_k]
        
        # context: [batch_size x n_heads x len_q x d_k]
        # attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s) # context是attn✖V
        # contiguous()的功能类似deepcopy
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_k) # context: [batch_size x len_q x n_heads * d_k] 最后一个维度是将8个head concat起来，维度依然512
        
        output = self.fc(context) # [batch_size, len_q, d_embed]
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    # 该模块也可用linear+ReLU实现
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embed, out_channels=d_hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_hidden, out_channels=d_embed, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_embed)
    def forward(self, x):
        residual = x # inputs : [batch_size, len_q, d_model]
        x = nn.ReLU()(self.conv1(x.transpose(1, 2)))
        x = self.conv2(x).transpose(1, 2)
        return self.layer_norm(x + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # 1个encoder layer由2个模块组成
        self.MultiHeadAttention = MultiHeadAttention()
        self.PoswiseFeedForwardNet = PoswiseFeedForwardNet()

    def forward(self, x):
        x, attn = self.MultiHeadAttention(x) # enc_inputs to same Q,K,V
        x = self.PoswiseFeedForwardNet(x) # enc_outputs: [batch_size x len_q x d_model]
        return x, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = Embedding(inpt_dim=d_obs, embed_dim=d_embed) # state dimension，embedding dimension
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_embed, d_classification)

    def forward(self, x): # enc_inputs : [batch_size x source_len]
        y = self.embedding(x)
        self_attns = []
        for layer in self.layers:
            y, self_attn = layer(y)
            self_attns.append(self_attn)
        
        y = y.mean(dim=1) # [batch_size, d_embed]
        out = torch.softmax(self.fc(y), dim=-1)
        return out, self_attns


encoder = Encoder()
for _ in range(10):
    trajectory = torch.rand(batch_size, len_traj, d_obs)
    pred, _ = encoder(trajectory)
    print(torch.argmax(pred,-1))


'''
from torchinfo import summary
summary(encoder, (batch_size, len_traj, d_obs))
print(context.shape, attn[0].shape)

'''
