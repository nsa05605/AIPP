#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:12:43 2020

@author: yongyongwei
"""

import torch.nn as nn
import torch.nn.functional as F

class ComaCritic(nn.Module):
    def __init__(self,input_shape, args):
        super(ComaCritic,self).__init__()
        self.args = args
        #config for GRU layer
        self.bidirectional = False
        self.num_layers = 1
        
        self.num_directions = 1 if self.bidirectional == False else 2
        """
        note we use GRU instead of GRUCell here
        input:[batch,seq_len, input_size]
        h0: [n_layers * num_directions, batch_size, hidden_size]
        ouput:[batch,seq_len, num_directions * hidden_size]
        hn: [n_layers * num_directions, batch_size, hidden_size]
        """
        self.rnn = nn.GRU(input_shape, args.critic_hidden_dim, num_layers = self.num_layers, batch_first=True,bidirectional = self.bidirectional)
        
        #note input is the output of the last step (equals to hn), i.e, output[:,-1,:] == h
        self.fc = nn.Linear(args.critic_hidden_dim * self.num_directions, args.n_actions)
        
    def forward(self,inputs):
        """
        output the q values (batch) for an agent 
        inputs size: <batch, seq_len, input_shape>
        output size: <batch, n_actions>
        """
        _, h_n = self.rnn(inputs)
        h = h_n.reshape(self.num_layers, self.num_directions, -1, self.args.critic_hidden_dim)
        h_in = h[-1].permute(1,0,2).reshape(-1,self.num_directions * self.args.critic_hidden_dim)
        out = self.fc(F.relu(h_in))
        return out
        
        
        