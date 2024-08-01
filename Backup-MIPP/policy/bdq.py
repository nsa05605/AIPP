#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:41:25 2020

@author: yongyongwei
"""

import torch
import os
from network.bdq_net import BDQCell

class ABR_BDQ:
    def __init__(self,args):
        #algs works with MARL with team reward
        assert args.alg.startswith('abr') == True, 'alg works as action branching'
        assert args.n_agents ==1
        self.args = args
        self.n_agents = args.n_agents 
        
        self.eval_rnn = BDQCell(args.nn_input_dim, args)
        self.target_rnn = BDQCell(args.nn_input_dim, args)
        
        self.eval_hidden = None
        self.target_hidden = None        
        
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        
        self.eval_parameters = list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr = args.lr)
        
        self.output_dir = os.path.join(args.output_dir, args.area_name, args.alg)
        subfolder = '{}-paths-charging-{}'.format(args.path_number,'-'.join([str(c) for c in args.chargingnodes]))
        self.output_dir = os.path.join(self.output_dir,subfolder)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)        
            
    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        episode_num = batch['s'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)   
        s, s_next, u, r, valid_u, valid_u_next, terminated = batch['s'], batch['s_next'],batch['u'],\
            batch['r'], batch['valid_u'],batch['valid_u_next'],batch['terminated']
    
        mask = 1 - batch['padded'].float()
        
        if self.args.cuda:
            u = u.cuda()
            valid_u = valid_u.cuda()
            valid_u_next = valid_u_next.cuda()
            r = r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()
            
        
        q_evals, q_targets = self.get_q_values(batch,max_episode_len)
        q_evals = torch.gather(q_evals, dim = 3, index = u).squeeze(3)
        q_targets [valid_u_next == 0] = -9999999
        q_targets = q_targets.max(dim = 3)[0]
        
         #Expand if team reward
        if self.args.n_robots > r.shape[-1]:
            r = r.expand(-1,-1,self.args.n_robots)
            terminated = terminated.expand(-1,-1, self.args.n_robots)
            mask = mask.repeat(1,1, self.args.n_robots)
        
        targets = r + self.args.gamma * (1 - terminated) * q_targets
        
        td_error = q_evals - targets.detach()
        masked_td_error = td_error * mask
        
        loss = (masked_td_error ** 2).sum()/mask.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        
        
    def get_q_values(self,batch,max_episode_len):
        episode_num = batch['s'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            nn_inputs = batch['s'][:,transition_idx,:].reshape(episode_num * self.n_agents, -1)
            nn_inputs_next = batch['s_next'][:,transition_idx,:].reshape(episode_num * self.n_agents, -1)
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                nn_inputs_next = nn_inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(nn_inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(nn_inputs_next, self.target_hidden)
            
            q_eval = torch.stack(q_eval,dim=1)
            q_target = torch.stack(q_target,dim=1)
            
            q_evals.append(q_eval)
            q_targets.append(q_target)
            
        q_evals = torch.stack(q_evals, dim = 1)
        q_targets = torch.stack(q_targets,dim=1)
                
        return q_evals, q_targets
        
        

    
    def init_hidden(self,episode_num):
        self.eval_hidden = self.eval_rnn.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents, -1)
        self.target_hidden = self.target_rnn.init_hidden().unsqueeze(0).expand(episode_num,self.n_agents,-1)
        
        
    def save_model(self,train_step):
        num = str(train_step // self.args.save_cycle)
        model_name = '{}_abr_rnn_rt{}.pkl'.format(num,self.args.reward_type)
        torch.save(self.eval_rnn.state_dict(), os.path.join(self.output_dir, model_name))