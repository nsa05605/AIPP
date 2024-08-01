#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 08 15:57:38 2020

@author: yongyongwei
"""
import torch.nn as nn
import torch.nn.functional as F


class BDQCell(nn.Module):
    def __init__(self,input_shape,args):
        super(BDQCell,self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnncell = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        self.linear_V = nn.Linear(args.rnn_hidden_dim, 1)
        
        self.branches = []
        for i in range(args.n_robots):
            self.branches.append(
                nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                              nn.ReLU(),
                              nn.Linear(args.rnn_hidden_dim, args.n_actions))
                )

    def init_hidden(self):
        return self.fc1.weight.new(1,self.args.rnn_hidden_dim).zero_()
    
    def forward(self,obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1,self.args.rnn_hidden_dim)
        h = self.rnncell(x,h_in)
        
        v = self.linear_V(h)
        out = []
        for i in range(self.args.n_robots):
            bout = self.branches[i](h)
            out.append(v + (bout - bout.mean(dim=-1,keepdim=True).expand_as(bout)))
        return out,h
    
    

    