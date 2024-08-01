#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:24:58 2020

@author: yongyongwei
"""

import torch
import os
from network.base_net import RNNCell
from network.can_net import CANNet

class ACCA:
    def __init__(self, args):
        #algs works with MARL with team reward
        assert args.alg.startswith('ma') == True, 'alg works as MARL with team reward'
        assert args.n_agents > 1
        assert args.n_agents == args.n_robots
        assert args.reward_type == 0, 'alg works as MARL with team reward'
        
        self.args = args
        self.n_agents = args.n_agents   
        
        #reuse network is a bad idea, not support anymore
        assert args.reuse_network == False
        assert args.full_observable == True
        
        self.eval_net = []
        self.eval_critic = []
        self.target_critic = []
        for agent_id in range(self.n_agents):
            self.eval_net.append(RNNCell(args.nn_input_dim, args))
            self.eval_critic.append(RNNCell(args.nn_input_dim, args,critic = True))
            self.target_critic.append(RNNCell(args.nn_input_dim, args,critic = True))
        
        self.eval_can_net = CANNet(args)

        self.eval_hidden = None
        self.eval_critic_hidden = None
        self.target_critic_hidden = None
        
        for agent_id in range(self.n_agents):
            self.target_critic[agent_id].load_state_dict(self.eval_critic[agent_id].state_dict())
        
        eval_net_parameters = []
        critic_parameters = []
        for agent_id in range(self.n_agents):
            eval_net_parameters.extend(list(self.eval_net[agent_id].parameters()))
            critic_parameters.extend(list(self.eval_critic[agent_id].parameters()))
        
        self.eval_parameters = list(self.eval_can_net.parameters()) + eval_net_parameters + critic_parameters
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr = args.lr)     
        
        self.output_dir = os.path.join(args.output_dir, args.area_name, args.alg)
        subfolder = '{}-paths-reuse_network-{}-full_observable-{}-charging-{}'.format(args.path_number,args.reuse_network,args.full_observable,'-'.join([str(c) for c in args.chargingnodes]))
        self.output_dir = os.path.join(self.output_dir,subfolder)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)          

        print('Init algorithm {}'.format(args.alg))
        
    def learn(self, batch, max_episode_len, train_step, epsilon = None,agent_id=None): 
        episode_num = batch['s'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)   
        u, r, valid_u, valid_u_next, terminated = batch['u'],\
            batch['r'], batch['valid_u'],batch['valid_u_next'],batch['terminated']
    
        mask = (1 - batch['padded'].float()).squeeze(-1)
        
        
        v_evals, v_next_targets = self._get_v_values(batch, max_episode_len)

        action_prob, hidden_evals = self._get_action_prob(batch, max_episode_len)
        
        pred_r = self.get_can(batch, hidden_evals)
        
        #1. Loss of credit assignment
        can_err = ((r - pred_r.sum(dim=-1, keepdim=True))).squeeze(-1) * mask
        l_can =  (can_err**2).sum()/mask.sum()
        
        #2. Loss of td error of state value
        v_targets = pred_r + v_next_targets.squeeze(-1)
        td_error = v_targets.detach() - v_evals.squeeze(-1)
        mask_all_agents = mask.unsqueeze(-1).expand(-1,-1,self.n_agents)
        l_td = ((td_error * mask_all_agents) ** 2).sum()/mask_all_agents.sum()
        
        
        #3. Loss of policy gradient
        pi_taken = torch.gather(action_prob, dim=-1, index = u).squeeze(-1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)
        l_pg = -((td_error.detach() * log_pi_taken) * mask_all_agents).sum()/mask_all_agents.sum()
        
        loss = l_can + l_td + l_pg

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        if train_step >0 and train_step % self.args.target_update_cycle == 0:
            print('loss of can:',l_can,'grad_norm:',grad_norm)
            for agent_id in range(self.n_agents):
                self.target_critic[agent_id].load_state_dict(self.eval_critic[agent_id].state_dict())                
                    

    
    def _get_action_prob(self,batch, max_episode_len):
        episode_num = batch['s'].shape[0]
        action_logits,  hidden_evals = [], []
        valid_actions = batch['valid_u']
        for transition_idx in range(max_episode_len):
            #exclude agent IDs
            nn_inputs = batch['s'][:,transition_idx,:,:3 * self.n_agents]
            logits_eval = []
            for agent_id in range(self.n_agents):
                logits_agent, self.eval_hidden[agent_id] = self.eval_net[agent_id](nn_inputs[:,agent_id,:],self.eval_hidden[agent_id])
                logits_eval.append(logits_agent)
    
            logits_eval = torch.stack(logits_eval,dim=1) 
            hidden_eval = torch.stack(self.eval_hidden, dim=1)

            logits_eval = logits_eval.reshape(episode_num, self.n_agents,-1) 
            hidden_eval = hidden_eval.reshape(episode_num, self.n_agents,-1)
          
            action_logits.append(logits_eval)
            hidden_evals.append(hidden_eval)

        action_logits = torch.stack(action_logits, dim = 1)
        hidden_evals = torch.stack(hidden_evals,dim=1)
        
        #mask invalid actions
        action_logits [valid_actions ==0] = -1e10
        action_prob = torch.nn.functional.softmax(action_logits, dim=-1)
       
        return action_prob, hidden_evals

    def _get_v_values(self,batch, max_episode_len):
        episode_num = batch['s'].shape[0]
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            #exclude agent IDs
            nn_inputs = batch['s'][:,transition_idx,:,:3 * self.n_agents]
            nn_inputs_next = batch['s_next'][:,transition_idx,:,:3 * self.n_agents]
            if transition_idx == 0:
                for agent_id in range(self.n_agents):
                    _, self.target_critic_hidden[agent_id] = self.target_critic[agent_id](nn_inputs[:,agent_id,:],self.eval_critic_hidden[agent_id])
            v_eval,v_target = [],[]
            for agent_id in range(self.n_agents):
                v_eval_agent, self.eval_critic_hidden[agent_id] = self.eval_critic[agent_id](nn_inputs[:,agent_id,:],self.eval_critic_hidden[agent_id])
                v_target_agent, self.target_critic_hidden[agent_id] = self.target_critic[agent_id](nn_inputs_next[:,agent_id,:],self.target_critic_hidden[agent_id])
                v_eval.append(v_eval_agent)
                v_target.append(v_target_agent)
            v_eval = torch.stack(v_eval,dim=1)
            v_target = torch.stack(v_target,dim=1)
            
            v_eval = v_eval.reshape(episode_num, self.n_agents,-1)
            v_target = v_target.reshape(episode_num, self.n_agents, -1)
            
            v_evals.append(v_eval)
            v_targets.append(v_target)

        v_evals = torch.stack(v_evals, dim = 1)
        v_targets= torch.stack(v_targets, dim=1)

        return v_evals, v_targets
    
    def get_can(self, batch, hidden_evals):
        episode_num, max_episode_len, n_agents, _ =  hidden_evals.shape
        u_onehot = batch['u_onehot'][:,:max_episode_len]
        pred_r = self.eval_can_net(hidden_evals, u_onehot)
        pred_r = pred_r.reshape(episode_num, max_episode_len, n_agents,1).squeeze(-1)
        return pred_r
        
    
    def init_hidden(self,episode_num):
        self.eval_hidden,self.eval_critic_hidden, self.target_critic_hidden = [],[],[]
        for agent_id in range(self.n_agents):
            self.eval_hidden.append(self.eval_net[agent_id].init_hidden().expand(episode_num, -1))
            self.eval_critic_hidden.append(self.eval_critic[agent_id].init_hidden().expand(episode_num, -1))
            self.target_critic_hidden.append(self.target_critic[agent_id].init_hidden().expand(episode_num, -1))
        
    def save_model(self,train_step, agent_id = None):
        num = str(train_step // self.args.save_cycle)
        for agent_id in range(self.n_agents):
            torch.save(self.eval_net[agent_id].state_dict(), os.path.join(self.output_dir, '{}_rnn-agent{}.pkl'.format(num,agent_id)))
        torch.save(self.eval_can_net.state_dict(), os.path.join(self.output_dir, '{}_can_params.pkl'.format(num)))
        
    def load_model(self,folder, num):
        for agent_id in range(self.n_agents):
            rnn_file_name = '{}/{}_rnn-agent{}.pkl'.format(folder,num,agent_id)
            self.eval_net[agent_id].load_state_dict(torch.load(rnn_file_name))
        can_file = '{}/{}_can_params.pkl'.format(folder,num)
        self.eval_can_net.load_state_dict(torch.load(can_file))
        
             