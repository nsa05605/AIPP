#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:35:45 2020

@author: yongyongwei
"""

import torch 
import os
from network.base_net import RNNCell,MLP
class DQN:
    def __init__(self,args):
        #force n_agents = 1, and runner will create multi-agent instances
        if args.n_agents > 1 and args.reuse_network == False:
            self.n_agents = 1
        else:
            self.n_agents = args.n_agents
        
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
        #sequential rollout case
        if args.n_agents == 1:
            subfolder = '{}-paths-charging-{}'.format(args.path_number,'-'.join([str(c) for c in args.chargingnodes]))
            self.output_dir = os.path.join(self.output_dir,subfolder)
        else:
            subfolder = '{}-paths-reuse_network-{}-full_observable-{}-charging-{}'.format(args.path_number,args.reuse_network,args.full_observable,'-'.join([str(c) for c in args.chargingnodes]))
            self.output_dir = os.path.join(self.output_dir,subfolder)
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def learn(self, batch, max_episode_len, train_step, epsilon=None, agent_id = None):
        if self.args.state_type ==0:  
            episode_num = batch['s'].shape[0]
            self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)   
        s, s_next, u, r, valid_u, valid_u_next, terminated = batch['s'], batch['s_next'],batch['u'],\
            batch['r'], batch['valid_u'],batch['valid_u_next'],batch['terminated']
    
        if self.args.state_type==0: 
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
            
        
        q_evals, q_targets = self.get_q_values(batch,max_episode_len, agent_id)

        #the following could be unified by using dim=-1, but I did not test it
        if self.args.state_type==0:
            q_evals = torch.gather(q_evals, dim = 3, index = u).squeeze(3)
            q_targets [valid_u_next == 0] = -9999999
            q_targets = q_targets.max(dim = 3)[0]
        else:
            q_evals = torch.gather(q_evals, dim = -1, index = u).squeeze(-1)
            q_targets [valid_u_next == 0] = -9999999
            q_targets = q_targets.max(dim = -1)[0]            
        
         #Expand if team reward
        if self.n_agents > r.shape[-1]:
            r = r.expand(-1,-1,self.n_agents)
            terminated = terminated.expand(-1,-1, self.n_agents)
            mask = mask.repeat(1,1, self.n_agents)
        
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
            #prepare for multi-agents, partial observe, reuse network case
            if self.args.n_agents > 1 and self.args.full_observable == False and self.args.reuse_network == True:
                agent_indices = torch.argmax(batch['s'][:,:,:,-self.args.n_agents:],dim=-1,keepdim=True)
                agent_indices = agent_indices.repeat(1,1,1,3) * 3 + torch.tensor(range(3))
                agent_feature = torch.gather(batch['s'][:,:,:,:3 * self.args.n_agents],dim=-1,index=agent_indices)
                agent_feature = torch.cat((agent_feature, batch['s'][:,:,:,3*self.args.n_agents:]),dim=-1)        
    
                agent_indices_next = torch.argmax(batch['s_next'][:,:,:,-self.args.n_agents:],dim=-1,keepdim=True)
                agent_indices_next = agent_indices_next.repeat(1,1,1,3) * 3 + torch.tensor(range(3))
                agent_feature_next = torch.gather(batch['s_next'][:,:,:,:3 * self.args.n_agents],dim=-1,index=agent_indices_next)
                agent_feature_next = torch.cat((agent_feature_next, batch['s_next'][:,:,:,3*self.args.n_agents:]),dim=-1) 
                
            for transition_idx in range(max_episode_len):
                if self.args.full_observable or self.args.n_agents == 1:
                    nn_inputs = batch['s'][:,transition_idx,:].reshape(episode_num * self.n_agents, -1)
                    nn_inputs_next = batch['s_next'][:,transition_idx,:].reshape(episode_num * self.n_agents, -1)
                else:
                    if self.args.reuse_network:
                        #here self.n_agents = self.args.n_agents > 1
                        assert batch['s'].shape[-1] == self.args.n_agents * 3 + self.args.n_agents 
                        assert self.args.n_agents == self.n_agents
                        nn_inputs = agent_feature[:, transition_idx].reshape(episode_num * self.n_agents, -1)    
                        nn_inputs_next = agent_feature_next[:,transition_idx].reshape(episode_num * self.n_agents, -1) 
                    else:
                        #here, network not reused, each agent a network
                        #thus, self.n_agents = 1 (forced to be 1, third dim) < self.args.n_agents 
                        feature_idx = range(agent_id * 3, agent_id * 3 + 3)
                        nn_inputs = batch['s'][:, transition_idx,:, feature_idx].reshape(episode_num * self.n_agents, -1)     
                        nn_inputs_next = batch['s_next'][:, transition_idx,:, feature_idx].reshape(episode_num * self.n_agents, -1)     
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
        if self.args.alg.startswith('ma'):
            if agent_id is None:
                #if reuse_network True
                model_name = '{}_rnn_rt{}.pkl'.format(num,self.args.reward_type)
            else:
                model_name = '{}_rnn_rt{}-agent{}.pkl'.format(num,self.args.reward_type,agent_id)
        else:
            model_name = '{}_rnn.pkl'.format(num)
        torch.save(self.eval_net.state_dict(), os.path.join(self.output_dir, model_name))
        
                 
    def load_model(self,num,agent_id=None):
        if self.args.alg.startswith('ma'):
            model_name = '{}/{}_rnn_rt{}-agent{}.pkl'.format(self.output_dir,num,self.args.reward_type,agent_id)
        else:
            model_name = '{}/{}_rnn.pkl'.format(self.output_dir,num)
        self.eval_net.load_state_dict(torch.load(model_name))
            
        
        