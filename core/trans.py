import numpy as np
import torch
import torch.nn as nn
len_traj = 13
batch = 10
d_obs = 6
d_embed = 7 # embedding dimension
n_heads = 8
d_k = 16
trajectory = torch.rand(batch, len_traj, d_obs)

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

    def forward(self, Q, K, V):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0) # 残差跨层连接
        
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_q x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_q x d_k]
        
        # context: [batch_size x n_heads x len_q x d_k], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s) # context是attn✖V
        # contiguous()的功能类似deepcopy
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_k) # context: [batch_size x len_q x n_heads * d_k] 最后一个维度是将8个head concat起来，维度依然512
        
        output = self.fc(context) # [batch, len_q, d_embed]
        print(residual.shape)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

# print(trajectory.shape)
em_model = Embedding(d_obs, d_embed)
attn_model = MultiHeadAttention()
x = em_model(trajectory)
context, attn = attn_model(x, x, x)
print(context.shape, attn.shape)