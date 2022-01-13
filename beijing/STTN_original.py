# MODEL CODE
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020

@author: wb
"""

import torch
import torch.nn as nn
# from GCN_models import GCN
# from One_hot_encoder import One_hot_encoder
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs

DEVICE = 'cuda:3'


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])




class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, in_features, out_features, adj, K, bias=True):
#     def __init__(self, nfeat, nhid, nclass, dropout, K):
#     def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.DEVICE = DEVICE
        self.K = K
        adj = np.array(adj.cpu())
        L_tilde = scaled_Laplacian(adj)
        self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(self.DEVICE) for i in cheb_polynomial(L_tilde, K)]
        
        
        self.in_channels = in_features
        self.out_channels = out_features
        
        self.Theta = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (B, N, C] --> (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
#         x = x.permute(0, 1, 3, 2)
        batch_size, num_of_vertices, in_channels = x.shape

           

        graph_signal = x

        output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

        for k in range(self.K):

            T_k = self.cheb_polynomials[k]  # (N,N)

            theta_k = self.Theta[k]  # (in_channel, out_channel)

            rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1) # （b, F_in, N) * (N, N) --> (b, F_in, N) --> (b, N, F_in)

            output = output + rhs.matmul(theta_k) # (b, N, F_in) * (F_in, F_out) --> (b, N, F_out) 


        
        result = F.relu(output)

        return result
    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, cheb_K, dropout):
        super(GCN, self).__init__()

        self.gc1 = cheb_conv(nfeat, nhid, adj, cheb_K)
        self.gc2 = cheb_conv(nhid, nclass, adj, cheb_K)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)    
    
    
'''
Position Encoding
有三种方式：
这是其中一种， onehot 方式，也就是需要用到t的这种
另外还有两种，分别为：
    Torch.nn.embedding
    Sincos    --- NIPS 17  Attention is All You Need

'''    
    

class One_hot_encoder(nn.Module):
    def __init__(self, embed_size, time_num = 288):
        super(One_hot_encoder, self).__init__()
        
        self.time_num = time_num
        self.I = nn.Parameter(torch.eye(time_num, time_num, requires_grad=True))
        self.onehot_Linear = nn.Linear(time_num, embed_size)     # 线性层改变one hot编码维度

    def forward(self, i, N = 25, T = 12):
    
        if i%self.time_num+T > self.time_num :
            o1 = self.I[i%self.time_num : , : ]
            o2 = self.I[0 : (i+T)%self.time_num, : ]
            onehot = torch.cat((o1, o2), 0)
        else:        
            onehot = self.I[i%self.time_num: i%self.time_num+T, : ]
        
        #onehot = onehot.repeat(N, 1, 1)   
        onehot = onehot.expand(N, T, self.time_num)
        onehot = self.onehot_Linear(onehot)
        return onehot

    
'''
Attention 基础代码

ScaledDotProductAttention  是通用的

解释dk：
数据进来的时候是B,N,T,C，做attention的时候，C=1 ，不能很好的表征数据高维空间的特征，C ---> embedded size 32 or 64  加入dk = 32，
那么一个头就是32，然后加上多头注意力机制的话，比如8个head，8个头，那就是32*8=256，如果要跟NIPS17 tranformer论文完全对应上，那么dk=64，head = 8 ，all embeded size = 512 

'''    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context
'''
S 代表spatial   ，MultiHeadAttention 代表多头注意力机制
'''
    
class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        # 用Linear来做投影矩阵    
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, N, T, C] --[B, N, T, self.heads, self.head_dim] ->  [B,h,T,N,dk]   然后是为了把N,dk这两维度考虑去做ScaledDotProductAttention ，代表着是spatial attention
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]  seq_len = N

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output

'''
T 代表Temporal   ，MultiHeadAttention 代表多头注意力机制
'''
    
class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        # 用Linear来做投影矩阵    
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) # Q: [B, h, N, T, d_k]   T，dk 就代表是temporal attention
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) #[B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output 



class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, cheb_K, dropout, forward_expansion):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.adj = adj
        self.D_S = adj.to(DEVICE)
        self.embed_liner = nn.Linear(adj.shape[0], embed_size)
        
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        # 调用GCN
        self.gcn = GCN(embed_size, embed_size*2, embed_size, adj, cheb_K, dropout)  
        self.norm_adj = nn.InstanceNorm2d(1)    # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # value, key, query: [N, T, C]  [B, N, T, C]        
        # Spatial Embedding 部分 
#         N, T, C = query.shape
#         D_S = self.embed_liner(self.D_S) # [N, C]
#         D_S = D_S.expand(T, N, C) #[T, N, C]相当于在第一维复制了T份
#         D_S = D_S.permute(1, 0, 2) #[N, T, C]
        B, N, T, C = query.shape
        D_S = self.embed_liner(self.D_S) # [N, C]    ---position encoding 
        D_S = D_S.expand(B, T, N, C) #[B, T, N, C] 相当于在第2维复制了T份, 第一维复制B份
        D_S = D_S.permute(0, 2, 1, 3) #[B, N, T, C]
        
        
        # GCN 部分


        X_G = torch.Tensor(B, N,  0, C).to(DEVICE)
        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)
        
        for t in range(query.shape[2]):
            o = self.gcn(query[ : ,:,  t,  : ],  self.adj) # [B, N, C]
            o = o.unsqueeze(2)              # shape [N, 1, C] [B, N, 1, C]
#             print(o.shape)
            X_G = torch.cat((X_G, o), dim=2)
         # 最后X_G [B, N, T, C]   
        
#         print('After GCN:')
#         print(X_G)
        # Spatial Transformer 部分
        query = query + D_S
        attention = self.attention(query, query, query) #(B, N, T, C)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        
        # 融合 STransformer and GCN  
        g = torch.sigmoid(self.fs(U_S) +  self.fg(X_G))      # (7)
        out = g*U_S + (1-g)*X_G                                # (8)

        return out #(B, N, T, C)    


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        
        # Temporal embedding One hot
        self.time_num = time_num
#         self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding


        
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t):
        B, N, T, C = query.shape
        
#         D_T = self.one_hot(t, N, T)                          # temporal embedding选用one-hot方式 或者
        D_T = self.temporal_embedding(torch.arange(0, T).to(DEVICE))    # temporal embedding选用nn.Embedding
        D_T = D_T.expand(B, N, T, C)


        # temporal embedding加到query。 原论文采用concatenated
        query = query + D_T  
        
        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out




### STBlock

class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, cheb_K, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, t):
    # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout
        x1 = self.norm1(self.STransformer(value, key, query) + query) #(B, N, T, C)
        x2 = self.dropout( self.norm2(self.TTransformer(x1, x1, x1, t) + x1) ) 
        return x2




### Encoder
class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        adj,
        time_num,
        device,
        forward_expansion,
        cheb_K,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    cheb_K,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
    # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)        
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out, t)
        return out     
    


### Transformer   
class Transformer(nn.Module):
    def __init__(
        self,
        adj,
        embed_size,
        num_layers,
        heads,
        time_num,
        forward_expansion, ##？
        cheb_K,
        dropout,
        
        device=DEVICE
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            cheb_K,
            dropout
        )
        self.device = device

    def forward(self, src, t): 
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src, t) 
        return enc_src # [B, N, T, C]


### ST Transformer: Total Model

class STTransformer(nn.Module):
    def __init__(
        self, 
        adj,
        in_channels, 
        embed_size, 
        time_num,
        num_layers,
        T_dim,
        output_T_dim,  
        heads,    
        cheb_K,
        forward_expansion,
        dropout = 0
    ):        
        super(STTransformer, self).__init__()

        self.forward_expansion = forward_expansion   # feed forward 的 embeded size  8，16，32....1024
        # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)  # Channel = 1 给 扩维，成 embeded size
        self.Transformer = Transformer(
            adj,
            embed_size, 
            num_layers, 
            heads, 
            time_num,
            forward_expansion,
            cheb_K,
            dropout = 0
        )

        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维   or 12in  12 out
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)  
        # 缩小通道数，降到1维。   
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()  #  和归一化搭配好，防止梯度爆炸，消失。
    
    def forward(self, x):
#         platform: (CHANNEL, TIMESTEP_IN, N_NODE)
        
        
        # input x shape[ C, N, T] 
        # C:通道数量。  N:传感器数量。  T:时间数量
        
#         x = x.unsqueeze(0)
        
#         x = np.transpose(x,(0,2,1)).to(DEVICE)
        input_Transformer = self.conv1(x)      # conv 要求第二维度是C，  也就是必须得B C +  其他     
#         input_Transformer = input_Transformer.squeeze(0)
#         input_Transformer = input_Transformer.permute(1, 2, 0)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)

        
        
        
        #input_Transformer shape[N, T, C]   [B, N, T, C]
        output_Transformer = self.Transformer(input_Transformer, self.forward_expansion)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)
        #output_Transformer shape[B, T, N, C]
        
#         output_Transformer = output_Transformer.unsqueeze(0)     
        out = self.relu(self.conv2(output_Transformer))    # 等号左边 out shape: [1, output_T_dim, N, C]        
        out = out.permute(0, 3, 2, 1)           # 等号左边 out shape: [B, C, N, output_T_dim]
        out = self.conv3(out)                   # 等号左边 out shape: [B, 1, N, output_T_dim]   
#         out = out.squeeze(1)
        out = out.permute(0,1,3,2)
#         print('out: ',out.shape)
        return out #[B, N, output_dim]
        # return out shape: [N, output_dim]


# def print_params(model_name, model):
#     param_count=0
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             param_count += param.numel()
#     print(f'{model_name}, {param_count} trainable parameters in total.')
#     return

# def main():
#     GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
#     device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
#     修改
#     model = GMAN(SE, TIMESTEP_IN, device).to(device)   修改
#     summary(model, [(TIMESTEP_IN, N_NODE),(TIMESTEP_IN+TIMESTEP_OUT, TE_DIM)], device=device)    修改
#     print_params('STTransformer',model)
    
# if __name__ == '__main__':
#     main()



    
