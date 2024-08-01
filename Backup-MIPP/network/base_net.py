#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:57:38 2020

@author: yongyongwei
"""
import torch.nn as nn
import torch.nn.functional as F


class RNNCell(nn.Module):
    def __init__(self,input_shape,args,critic=False):
        super(RNNCell,self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnncell = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        if critic == False:
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        else:
            self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
    def init_hidden(self):
        return self.fc1.weight.new(1,self.args.rnn_hidden_dim).zero_()
    
    def forward(self,obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1,self.args.rnn_hidden_dim)
        h = self.rnncell(x,h_in)
        q = self.fc2(h)
        
        return q,h
    
    
    
    
class ABRCell(nn.Module):
    def __init__(self,input_shape, args, critic = False):
        super(ABRCell,self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnncell = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        self.branches = []
        if critic == False:
            outdim = args.n_actions
        else:
            outdim = 1
        for i in range(args.n_robots):
            self.branches.append(
                nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                              nn.ReLU(),
                              nn.Linear(args.rnn_hidden_dim, outdim))
                )

    def init_hidden(self):
        return self.fc1.weight.new(1,self.args.rnn_hidden_dim).zero_()
    
    def forward(self,obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1,self.args.rnn_hidden_dim)
        h = self.rnncell(x,h_in)
        
        out = []
        for i in range(self.args.n_robots):
            bout = self.branches[i](h)
            out.append(bout)
        return out,h

class MLP(nn.Module):
    def __init__(self,input_shape, args, critic = False):
        super(MLP, self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(input_shape, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        
        if critic == False:
            self.out = nn.Linear(args.mlp_hidden_dim,  args.n_actions)
        else:
            self.out = nn.Linear(args.mlp_hidden_dim,  1)
        
    def forward(self, nn_inputs):
        h1 = F.relu(self.fc1(nn_inputs))
        h2 = F.relu(self.fc2(h1))
        return self.out(h2)
        
        
        
    