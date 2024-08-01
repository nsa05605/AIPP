#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:34:14 2020

@author: yongyongwei
"""

import torch 
import os
from network.base_net import RNNCell,MLP
class JAL:
    def __init__(self,args):
        #force n_agents = 1, and runner will create multi-agent instances
        self.n_agents = args.n_agents
        assert self.n_agents == 1
        
        self.args = args
        if args.state_type ==0:
            self.eval_net = RNNCell(args.nn_input_dim, args)
            self.target_rnn = RNNCell(args.nn_input_dim, args)
        else:
            self.eval_net = MLP(args.nn_input_dim, args)
            self.target_rnn = MLP(args.nn_input_dim, args)
        
        if self.args.cuda:
            self.eval_net.cuda()
            self.target_rnn.cuda()
            
        self.target_rnn.load_state_dict(self.eval_net.state_dict())
        
        self.eval_parameters = list(self.eval_net.parameters())
        
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr = args.lr)
        
        self.eval_hidden = None
        self.target_hidden = None
        
        self.output_dir = os.path.join(args.output_dir, args.area_name, args.alg)
        subfolder = '{}-paths-charging-{}'.format(args.path_number,'-'.join([str(c) for c in args.chargingnodes]))
        self.output_dir = os.path.join(self.output_dir,subfolder)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def learn(self, batch, max_episode_len, train_step, epsilon=None, agent_id = None):
        if self.args.state_type ==0:  
            episode_num = batch['s'].shape[0]
            self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u' or key =='u_index':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)   
        s, s_next, u, r, valid_u, valid_u_next, terminated = batch['s'], batch['s_next'],batch['u'],\
            batch['r'], batch['valid_u'],batch['valid_u_next'],batch['terminated']
    
        if self.args.state_type ==0:  
            mask = 1 - batch['padded'].float()
        else:
            mask = None
        
        if self.args.cuda:
            u = u.cuda()
            valid_u = valid_u.cuda()
            valid_u_next = valid_u_next.cuda()
            r = r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()
       
        
        u_index = batch['u_index'].unsqueeze(-1)
        q_evals, q_targets = self.get_q_values(batch,max_episode_len, agent_id)
        
        if self.args.state_type==0:
            q_evals = torch.gather(q_evals, dim = 3, index = u_index).squeeze(3)
            q_targets [valid_u_next == 0] = -9999999
            q_targets = q_targets.max(dim = 3)[0]
        else:
            q_evals = torch.gather(q_evals, dim = -1, index = u_index).squeeze(-1)
            q_targets [valid_u_next == 0] = -9999999
            q_targets = q_targets.max(dim = -1)[0]  
            
        assert self.n_agents == 1
        
        if self.args.state_type==0:
            targets = r + self.args.gamma * (1 - terminated) * q_targets
            td_error = q_evals - targets.detach()
            masked_td_error = td_error * mask
            loss = (masked_td_error ** 2).sum()/mask.sum()
        else:
            targets = r + self.args.gamma * (1 - terminated) * q_targets        
            td_error = q_evals - targets.detach()
            loss = (td_error ** 2).mean()     
        
        
        
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_net.state_dict())
        
        
    def get_q_values(self,batch,max_episode_len, agent_id):
        if self.args.state_type == 0:
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
                q_eval, self.eval_hidden = self.eval_net(nn_inputs, self.eval_hidden)
                q_target, self.target_hidden = self.target_rnn(nn_inputs_next, self.target_hidden)
                
                q_eval = q_eval.reshape(episode_num, self.n_agents,-1)
                q_target = q_target.reshape(episode_num, self.n_agents, -1)
                q_evals.append(q_eval)
                q_targets.append(q_target)
            q_evals = torch.stack(q_evals, dim = 1)
            q_targets = torch.stack(q_targets,dim=1)
        else:
            batch_size = batch['s'].shape[0]
            nn_inputs = torch.cat((batch['nodes_info'],batch['s']),dim=-1).reshape(batch_size * self.n_agents, -1)
            nn_inputs_next = torch.cat((batch['nodes_info_next'],batch['s_next']),dim=-1).reshape(batch_size * self.n_agents, -1)
            q_eval = self.eval_net(nn_inputs)
            q_target = self.target_rnn(nn_inputs_next)
            q_eval = q_eval.reshape(batch_size, self.n_agents, -1)
            q_target = q_target.reshape(batch_size, self.n_agents,-1)
            return q_eval.squeeze(-1), q_target.squeeze(-1)             
                
        return q_evals, q_targets
        
        
    def init_hidden(self,episode_num):
        self.eval_hidden = self.eval_net.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents, -1)
        self.target_hidden = self.target_rnn.init_hidden().unsqueeze(0).expand(episode_num,self.n_agents,-1)
        
        
    def save_model(self,train_step,agent_id=None):
        num = str(train_step // self.args.save_cycle)
        model_name = '{}_rnn.pkl'.format(num)
        torch.save(self.eval_net.state_dict(), os.path.join(self.output_dir, model_name))
        
    def load_model(self,num,agent_id=None):
        model_name = '{}/{}_rnn.pkl'.format(self.output_dir,num)
        self.eval_net.load_state_dict(torch.load(model_name))                 