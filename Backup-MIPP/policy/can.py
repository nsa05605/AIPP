#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:39:15 2020

@author: yongyongwei
"""

import torch
import os
from network.base_net import RNNCell
from network.can_net import CANNet

class CAN:
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
        self.target_rnn = []
        for agent_id in range(self.n_agents):
            self.eval_net.append(RNNCell(args.nn_input_dim, args))
            self.target_rnn.append(RNNCell(args.nn_input_dim, args))   
            
        self.eval_can_net = CANNet(args)

        self.eval_hidden = None
        self.target_hidden = None
        

        for agent_id in range(self.n_agents):
            self.target_rnn[agent_id].load_state_dict(self.eval_net[agent_id].state_dict())
        
        eval_net_parameters = []
        for agent_id in range(self.n_agents):
            eval_net_parameters.extend(list(self.eval_net[agent_id].parameters()))
        
        self.eval_parameters = list(self.eval_can_net.parameters()) + eval_net_parameters
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr = args.lr)     
        
        #self.can_parameters = list(self.eval_can_net.parameters())
        #self.can_optimizer = torch.optim.RMSprop(self.can_parameters, lr = args.lr)
        
    
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
        
              

        ind_q_evals, ind_q_targets, hidden_evals, hidden_targets = self._get_individual_q(batch, max_episode_len)
        
        
        ind_q_clone = ind_q_evals.clone()
        ind_q_clone[valid_u == 0] = -999999
        
        ind_q_targets[valid_u_next == 0] = -999999
        
        opt_onehot_eval = torch.zeros(*ind_q_clone.shape)
        opt_action_eval = ind_q_clone.argmax(dim = 3, keepdim=True)
        opt_onehot_eval = opt_onehot_eval.scatter(-1, opt_action_eval[:,:].cpu(), 1)
        
        opt_onehot_target = torch.zeros(*ind_q_targets.shape)
        opt_action_target = ind_q_targets.argmax(dim=3, keepdim=True)
        opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:,:].cpu(),1)
        
        pred_r = self.get_can(batch, hidden_evals)
  
        #Loss of credit assignment
        can_err = ((r - pred_r.sum(dim=-1, keepdim=True))).squeeze(-1) * mask
        l_can =  (can_err**2).sum()/mask.sum()
        
        #Loss of individual q
        y_indq =  pred_r + self.args.gamma * torch.gather(ind_q_targets, dim=-1, index=opt_action_target).squeeze(-1) * (1 - terminated)
        
        td_err = torch.gather(ind_q_evals, dim=-1, index = u).squeeze(-1) - y_indq.detach()
        l_indq = ((td_err.squeeze(-1) * mask.unsqueeze(-1)) ** 2).sum()/mask.unsqueeze(-1).sum()
        loss = self.args.can_lambda * l_can + l_indq
        """
        y_dqn = r.squeeze(-1) + self.args.gamma * joint_q_targets*(1 - terminated.squeeze(-1))
        td_error = joint_q_evals - y_dqn.detach()
        l_td = ((td_error * mask) ** 2).sum()/mask.sum()
        
        #L_opt
        q_sum_opt = ind_q_clone.max(dim=-1)[0].sum(dim=-1)
        joint_q_hat_opt, _ = self.get_qtran(batch,hidden_evals, hidden_targets, opt_onehot_eval, hat=True)
        
        opt_err = q_sum_opt - joint_q_hat_opt.detach()
        
        l_opt = ((opt_err * mask) ** 2).sum()/mask.sum()
        
        #L_nopt
        q_individual = torch.gather(ind_q_evals, dim=-1, index = u).squeeze(-1)
        q_sum_nopt = q_individual.sum(dim=-1)
        nopt_error = q_sum_nopt - joint_q_evals.detach()
        
        nopt_error = nopt_error.clamp(max=0)
        
        l_nopt = ((nopt_error * mask) ** 2).sum()/mask.sum()
        
        loss = l_td + self.args.lambda_opt * l_opt + self.args.lambda_nopt * l_nopt
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        
        if train_step >0 and train_step % self.args.target_update_cycle == 0:
            print('loss of can:',l_can,'grad_norm:',grad_norm)
            for agent_id in range(self.n_agents):
                self.target_rnn[agent_id].load_state_dict(self.eval_net[agent_id].state_dict())                
                    

    
    def _get_individual_q(self,batch, max_episode_len):
        episode_num = batch['s'].shape[0]
        q_evals, q_targets, hidden_evals, hidden_targets = [], [], [], []
        for transition_idx in range(max_episode_len):
            #exclude agent IDs
            nn_inputs = batch['s'][:,transition_idx,:,:3 * self.n_agents]
            nn_inputs_next = batch['s_next'][:,transition_idx,:,:3 * self.n_agents]
            if transition_idx == 0:
                for agent_id in range(self.n_agents):
                    _, self.target_hidden[agent_id] = self.target_rnn[agent_id](nn_inputs[:,agent_id,:],self.eval_hidden[agent_id])
        
            q_eval,q_target = [],[]
            for agent_id in range(self.n_agents):
                q_eval_agent, self.eval_hidden[agent_id] = self.eval_net[agent_id](nn_inputs[:,agent_id,:],self.eval_hidden[agent_id])
                q_target_agent, self.target_hidden[agent_id] = self.target_rnn[agent_id](nn_inputs_next[:,agent_id,:],self.target_hidden[agent_id])
                q_eval.append(q_eval_agent)
                q_target.append(q_target_agent)
            q_eval = torch.stack(q_eval,dim=1)
            q_target = torch.stack(q_target,dim=1)
            
            hidden_eval = torch.stack(self.eval_hidden, dim=1)
            hidden_target = torch.stack(self.target_hidden, dim = 1)
            
            q_eval = q_eval.reshape(episode_num, self.n_agents,-1)
            q_target = q_target.reshape(episode_num, self.n_agents, -1)
            
            hidden_eval = hidden_eval.reshape(episode_num, self.n_agents,-1)
            hidden_target = hidden_target.reshape(episode_num, self.n_agents, -1)
            
            q_evals.append(q_eval)
            q_targets.append(q_target)
            hidden_evals.append(hidden_eval)
            hidden_targets.append(hidden_target)
        q_evals = torch.stack(q_evals, dim = 1)
        q_targets= torch.stack(q_targets, dim=1)
        hidden_evals = torch.stack(hidden_evals,dim=1)
        hidden_targets = torch.stack(hidden_targets, dim=1)
        
        return q_evals, q_targets, hidden_evals, hidden_targets

    def get_can(self, batch, hidden_evals):
        episode_num, max_episode_len, n_agents, _ =  hidden_evals.shape
        u_onehot = batch['u_onehot'][:,:max_episode_len]
        pred_r = self.eval_can_net(hidden_evals, u_onehot)
        pred_r = pred_r.reshape(episode_num, max_episode_len, n_agents,1).squeeze(-1)
        return pred_r
        
    
    def init_hidden(self,episode_num):
        self.eval_hidden,self.target_hidden = [],[]
        for agent_id in range(self.n_agents):
            self.eval_hidden.append(self.eval_net[agent_id].init_hidden().expand(episode_num, -1))
            self.target_hidden.append(self.target_rnn[agent_id].init_hidden().expand(episode_num, -1))
        
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
        
             