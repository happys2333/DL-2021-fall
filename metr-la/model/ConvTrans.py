import torch
import torch.nn as nn
from torchsummary import summary
# DEVICE = 'cuda:1'

import torch
import torch.nn.functional as F


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, in_channels=1, embedding_size=256, k=5):
        super(context_embedding, self).__init__()
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size=k)

    def forward(self, x):
        x = self.causal_convolution(x)
        return torch.tanh(x)


# !/usr/bin/env python
# coding: utf-8


class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper

    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    """

    def __init__(self,device,channel,input_step,dmodel):
        super(TransformerTimeSeries, self).__init__()
        self.input_embedding = context_embedding(channel, dmodel, 9)

        self.positional_embedding = nn.Embedding(input_step, dmodel) 

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=dmodel, nhead=8,dim_feedforward=32,activation='relu' )
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)
        self.device = device
        self.dmodel = dmodel
        self.fc1 = torch.nn.Linear(dmodel, 1)
        self.conv1 = nn.Conv2d(input_step, input_step, 1)
        self.conv2 = nn.Conv2d(dmodel, 1, 1)

    def forward(self, x):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        # x BCNT
        # z (BCT)
        b,c,n,t = x.shape
        x = x.permute(0,2,3,1)
        z = x.reshape(b*n,t,c).permute(0,2,1)


        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)
#         r = z[:,0,:]
#         print('r shape is :',r.shape)
#         print('z_embedding shape :',z_embedding.shape)  # [12, 414, 256]) [T,*N,C]
        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
#         positional_embeddings = self.positional_embedding(r.type(torch.long)).permute(1, 0, 2)  #  .type(torch.long)
#         print('puits: ',self.positional_embedding(torch.arange(0, t).to(self.device)).shape)
        positional_embeddings = self.positional_embedding(torch.arange(0, t).to(self.device)).expand(b*n,t,self.dmodel).permute(1,0,2)
        
        input_embedding = z_embedding + positional_embeddings
        #         input_embedding = input_embedding.type(torch.double)

#         print('positional_embeddings shape :',positional_embeddings.shape)
        
        input_embedding= input_embedding.permute(1,0,2)  
#         print('input_embedding shape :',input_embedding.shape)
        
        transformer_embedding = self.transformer_decoder(input_embedding) # [b*n, t, 256])
        
        out = transformer_embedding.reshape(b,n,t,self.dmodel)
        out = self.conv1(out.permute(0,2,1,3)) #(b,n,t,self.dmodel) - > [b,t,n,dmodel] ->  [b,t_out,n,dmodel]
        out = self.conv2(out.permute(0,3,1,2)) # [b,t_out,n,dmodel] -> [b,dmodel,t_out,n] -> [b,1,t_out,n]
#         print('transformer_embedding shape :',transformer_embedding.shape)
#         output = self.fc1(transformer_embedding.permute(1, 0, 2))

#         out = output.reshape(b,n,t,c)
#         out = out.permute(0,3,2,1)
        
        
#         print('out shape :',out.shape)        
        #back BCTN
        return out
import pandas as pd
from Param import *
def print_params(model_name, model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return
import sys
def main():
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '1'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")

    in_channels = 1  # Channels of input
    embed_size = 256  # Dimension of hidden embedding features
    kernel_width = 9  # Order for Chebyshev Polynomials (Eq 2)
    hidden_size = 512

    ### Construct Network
    model = TransformerTimeSeries(device,channel=in_channels,input_step=12,dmodel=256).to(device)

    summary(model, (CHANNEL, N_NODE, TIMESTEP_IN),device=device)
    print_params('LSTM', model)


if __name__ == '__main__':
    main()