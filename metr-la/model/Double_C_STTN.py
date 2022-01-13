# from folder workMETRLA

# MODEL CODE
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020

@author: wb
"""

import torch
import torch.nn as nn
import math

# from GCN_models import GCN
# from One_hot_encoder import One_hot_encoder
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs
from Param import *
from torchsummary import summary

DEVICE = 'cuda:1'


class One_hot_encoder(nn.Module):
    def __init__(self, embed_size, time_num=288):
        super(One_hot_encoder, self).__init__()

        self.time_num = time_num
        self.I = nn.Parameter(torch.eye(time_num, time_num, requires_grad=True))
        self.onehot_Linear = nn.Linear(time_num, embed_size)  # 线性层改变one hot编码维度

    def forward(self, i, N=25, T=12):

        if i % self.time_num + T > self.time_num:
            o1 = self.I[i % self.time_num:, :]
            o2 = self.I[0: (i + T) % self.time_num, :]
            onehot = torch.cat((o1, o2), 0)
        else:
            onehot = self.I[i % self.time_num: i % self.time_num + T, :]

        # onehot = onehot.repeat(N, 1, 1)
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
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
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
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1,
                                                                                 3)  # Q: [B, N, T, C] --[B, N, T, self.heads, self.head_dim] ->  [B,h,T,N,dk]   然后是为了把N,dk这两维度考虑去做ScaledDotProductAttention ，代表着是spatial attention
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]  seq_len = N

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
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
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2,
                                                                               4)  # Q: [B, h, N, T, d_k]   T，dk 就代表是temporal attention
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
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
        self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化

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
        D_S = self.embed_liner(self.D_S)  # [N, C]    ---position encoding
        D_S = D_S.expand(B, T, N, C)  # [B, T, N, C] 相当于在第2维复制了T份, 第一维复制B份
        D_S = D_S.permute(0, 2, 1, 3)  # [B, N, T, C]

        # Spatial Transformer 部分
        query = query + D_S
        attention = self.attention(query, query, query)  # (B, N, T, C)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        # 融合 STransformer and GCN
        g = torch.sigmoid(self.fs(U_S))  # (7)
        out = g * U_S + (1 - g)  # (8)

        return out  # (B, N, T, C)


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
        D_T = self.temporal_embedding(torch.arange(0, T).to(DEVICE))  # temporal embedding选用nn.Embedding
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
        x1 = self.norm1(self.STransformer(value, key, query) + query)  # (B, N, T, C)
        x2 = self.dropout(self.norm2(self.TTransformer(x1, x1, x1, t) + x1))
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
            forward_expansion,  ##？
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
        return enc_src  # [B, N, T, C]


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
            dropout=0
    ):
        super(STTransformer, self).__init__()

        self.forward_expansion = forward_expansion  # feed forward 的 embeded size  8，16，32....1024
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
            dropout=0
        )

        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维   or 12in  12 out
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, in_channels, 1)
        self.relu = nn.ReLU()  # 和归一化搭配好，防止梯度爆炸，消失。

    def forward(self, x):
        #         platform: (CHANNEL, TIMESTEP_IN, N_NODE)

        # input x shape[ C, N, T]
        # C:通道数量。  N:传感器数量。  T:时间数量

        #         x = x.unsqueeze(0)

        #         x = np.transpose(x,(0,2,1)).to(DEVICE)
        input_Transformer = self.conv1(x)  # conv 要求第二维度是C，  也就是必须得B C +  其他
        #         input_Transformer = input_Transformer.squeeze(0)
        #         input_Transformer = input_Transformer.permute(1, 2, 0)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)

        # input_Transformer shape[N, T, C]   [B, N, T, C]
        output_Transformer = self.Transformer(input_Transformer, self.forward_expansion)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)
        # output_Transformer shape[B, T, N, C]

        #         output_Transformer = output_Transformer.unsqueeze(0)
        out = self.relu(self.conv2(output_Transformer))  # 等号左边 out shape: [1, output_T_dim, N, C]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim]
        out = self.conv3(out)  # 等号左边 out shape: [B, 1, N, output_T_dim]
        #         out = out.squeeze(1)
        out = out.permute(0, 1, 3, 2)
        #         print('out: ',out.shape)
        return out  # [B, N, output_dim]
        # return out shape: [N, output_dim]


def print_params(model_name, model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return


import sys
import pandas as pd


def main():
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '1'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    in_channels = 2  # Channels of input
    embed_size = 32  # Dimension of hidden embedding features
    time_num = 288
    num_layers = 2  # Number of ST Block
    T_dim = 12  # Input length, should be the same as prepareData.py
    output_T_dim = 12  # Output Expected length
    heads = 4  # Number of Heads in MultiHeadAttention
    cheb_K = 2  # Order for Chebyshev Polynomials (Eq 2)
    forward_expansion = 32  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    dropout = 0
    A = pd.read_csv(ADJPATH).values
    A = torch.Tensor(A)
    ### Construct Network
    model = STTransformer(
        A,
        in_channels,
        embed_size,
        time_num,
        num_layers,
        T_dim,
        output_T_dim,
        heads,
        cheb_K,
        forward_expansion,
        dropout).to(DEVICE)

    summary(model, (2, N_NODE, TIMESTEP_IN), device=device)
    print_params('STTransformer', model)


if __name__ == '__main__':
    main()

'''

布置作业：

1. 设计  only  Spatial Transformer 的版本，跑出PEMSBAY的结果   12 步in  12 步 out
2. 设计  only  Temporal Transformer 的版本，跑出PEMSBAY的结果   12 步in  12 步 out
3. 设计  Temporal-Spatial Transformer 的版本，跑出PEMSBAY的结果   12 步in  12 步 out

4. 前面的版本完成后，全部升级为，C 维度由1变成2，多的一个C是时间戳，时间戳的写法，参考
也就是说原来是B N T C=1  ，现在要求改成  B,N,T,C=2， 然后跑出1，2，3 升级版结果。  12 步in  12 步 out   PEMSBAY 数据集

'''

