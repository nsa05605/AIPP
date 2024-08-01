#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:55:22 2020

@author: yongyongwei
"""

import torch 
import os
from network.base_net import RNNCell

class Reinforce:
    def __init__(self,args):
        #force n_agents = 1, and runner will create multi-agent instances
        if args.n_agents > 1 and args.reuse_network == False:
            self.n_agents = 1
        else:
            self.n_agents = args.n_agents
            
        self.args = args
        self.eval_rnn = RNNCell(args.nn_input_dim, args)
        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr = args.lr_actor)
        self.eval_hidden = None
        
        
        if self.args.cuda:
            self.eval_rnn.cuda()
            
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
            
    def learn(self, batch, max_episode_len, train_step, epsilon, agent_id = None):
        episode_num = batch['s'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)
        
        u, r, valid_u, terminated = batch['u'], batch['r'], batch['valid_u'],batch['terminated']
        mask = 1 - batch['padded'].float()
        if self.args.cuda:
            r = r.cuda()
            u = u.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()
            
        n_return = self._get_returns(r, mask, terminated, max_episode_len)
        
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon, agent_id)
        
        if self.n_agents > mask.shape[-1]:
            mask = mask.repeat(1,1,self.n_agents)
            
        pi_taken = torch.gather(action_prob, dim = 3, index = u).squeeze(3)
        pi_taken[mask ==0] = 1.0
        log_pi_taken = torch.log(pi_taken)
        
        loss =  - ((n_return * log_pi_taken) * mask).sum()/mask.sum()
        
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        #print('Actor loss is ',loss)

            
    def _get_returns(self, r, mask, terminated, max_episode_len):
        #terminated is not needed
        #terminated = 1 - terminated
        #ref https://github.com/starry-sky6688/StarCraft/issues/21#issuecomment-624603906
        n_return = torch.zeros_like(r)
        n_return[:,-1,:] = r[:,-1,:] * mask[:,-1,:]
        for transition_idx in range(max_episode_len - 2, -1, -1):
            n_return[:,transition_idx,:] = (r[:,transition_idx,:] + 
                                            self.args.gamma * n_return[:, transition_idx+1,:]) * mask[:,transition_idx,:]
        #Expand if team reward
        if self.n_agents > r.shape[-1]:
            n_return = n_return.expand(-1,-1,self.n_agents)
            
        return n_return
    
    def _get_action_prob(self, batch, max_episode_len, epsilon, agent_id):
        #here valid actions is one-hot encoded instead of action number
        episode_num = batch['s'].shape[0]
        valid_actions = batch['valid_u']
        action_logits = []
        #prepare for multi-agents, partial observe, reuse network case
        if self.args.n_agents > 1 and self.args.full_observable == False and self.args.reuse_network == True:
            agent_indices = torch.argmax(batch['s'][:,:,:,-self.args.n_agents:],dim=-1,keepdim=True)
            agent_indices = agent_indices.repeat(1,1,1,3) * 3 + torch.tensor(range(3))
            agent_feature = torch.gather(batch['s'][:,:,:,:3 * self.args.n_agents],dim=-1,index=agent_indices)
            agent_feature = torch.cat((agent_feature, batch['s'][:,:,:,3*self.args.n_agents:]),dim=-1)        
        for transition_idx in range(max_episode_len):
            if self.args.full_observable or self.args.n_agents == 1:
                nn_inputs = batch['s'][:,transition_idx,:].reshape(episode_num * self.n_agents, -1)
            else:
                if self.args.reuse_network:
                    #here self.n_agents = self.args.n_agents > 1
                    assert batch['s'].shape[-1] == self.args.n_agents * 3 + self.args.n_agents 
                    assert self.args.n_agents == self.n_agents
                    nn_inputs = agent_feature[:, transition_idx].reshape(episode_num * self.n_agents, -1)
                else:
                    #here, network not reused, each agent a network
                    #thus, self.n_agents = 1 (forced to be 1, third dim) < self.args.n_agents 
                    feature_idx = range(agent_id * 3, agent_id * 3 + 3)
                    nn_inputs = batch['s'][:, transition_idx,:, feature_idx].reshape(episode_num * self.n_agents, -1)                    
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            #the forward function of rnn will reshape eval_hidden
            outputs, self.eval_hidden = self.eval_rnn(nn_inputs, self.eval_hidden)
            outputs = outputs.reshape(episode_num, self.n_agents, -1)
            action_logits.append(outputs)
        action_logits = torch.stack(action_logits,dim = 1).cpu()
        action_logits[valid_actions == 0] = -1e10
        action_prob = torch.nn.functional.softmax(action_logits, dim = -1)
        if self.args.softmax_noise:
            action_num = valid_actions.sum(dim = -1, keepdim = True)
            #action_num can be 0 for padded transition and actions, thus the below can be inf
            action_prob = (1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num
            #set the prob of invalid actions to 0
            action_prob[valid_actions == 0] = 0
            
        if self.args.cuda:
            action_prob = action_prob.cuda()
            
        return action_prob
        
    def init_hidden(self, episode_num):
        self.eval_hidden = self.eval_rnn.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents,-1)
    
    
    def save_model(self, train_step,agent_id=None):
        num = str(train_step // self.args.save_cycle)
        if self.args.alg.startswith('ma'):
            if agent_id is None:
                #if reuse_network True
                model_name = '{}_rnn_rt{}.pkl'.format(num,self.args.reward_type)
            else:
                model_name = '{}_rnn_rt{}-agent{}.pkl'.format(num,self.args.reward_type,agent_id)
        else:
            model_name = '{}_rnn.pkl'.format(num)
        torch.save(self.eval_rnn.state_dict(), os.path.join(self.output_dir, model_name))
        
        