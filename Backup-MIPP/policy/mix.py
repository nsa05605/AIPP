#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:10:35 2020

@author: yongyongwei
"""

import torch
import os
from network.base_net import RNNCell,ABRCell
from network.mix_net import VDNNet, QMixNet
#import pdb
class MIX:
    def __init__(self,args):
        #algs works with MARL with team reward
        assert args.alg.startswith('ma') == True, 'alg works as MARL with team reward'
        assert args.n_agents > 1
        assert args.n_agents == args.n_robots
        assert args.reward_type == 0, 'alg works as MARL with team reward'
        
        self.args = args
        self.n_agents = args.n_agents 
        assert self.n_agents > 1

        if args.reuse_network == True:        
            self.eval_net = RNNCell(args.nn_input_dim, args)
            self.target_rnn = RNNCell(args.nn_input_dim, args)
        else:
            self.eval_net = []
            self.target_rnn = []
            for agent_id in range(self.n_agents):
                self.eval_net.append(RNNCell(args.nn_input_dim, args))
                self.target_rnn.append(RNNCell(args.nn_input_dim, args))
        
        self.eval_hidden = None
        self.target_hidden = None

        if args.mixer == 'qmix':
            #The global state (do not include one_hot agent ID information)
            self.hyper_input_dim = 3 * self.n_agents
            self.eval_mixer = QMixNet(self.hyper_input_dim, args)
            self.target_mixer = QMixNet(self.hyper_input_dim, args)
            self.eval_mixer_hidden = None
            self.target_mixer_hidden = None
            
        elif args.mixer == 'vdn':
            self.eval_mixer = VDNNet()
            self.target_mixer = VDNNet()
        else:
            raise ValueError('Mixer {} not recognised'.format(args.mixer))
            
        if self.args.cuda:
            if args.reuse_network == True:   
                self.eval_net.cuda()
                self.target_rnn.cuda()
            else:
                for agent_id in range(self.n_agents):
                    self.eval_net[agent_id].cuda()
                    self.target_rnn[agent_id].cuda()
                    
            self.eval_mixer.cuda()
            self.target_mixer.cuda()
        
        if args.reuse_network == True: 
            self.target_rnn.load_state_dict(self.eval_net.state_dict())
        else:
            for agent_id in range(self.n_agents):
                self.target_rnn[agent_id].load_state_dict(self.eval_net[agent_id].state_dict())
                
        self.target_mixer.load_state_dict(self.eval_mixer.state_dict())

        if args.reuse_network == True:  
            self.eval_parameters = list(self.eval_mixer.parameters()) + list(self.eval_net.parameters())
        else:
            eval_net_parameters = []
            for agent_id in range(self.n_agents):
                eval_net_parameters.extend(list(self.eval_net[agent_id].parameters()))
            self.eval_parameters = list(self.eval_mixer.parameters()) + eval_net_parameters
        
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr = args.lr)        
        
        self.output_dir = os.path.join(args.output_dir, args.area_name, args.alg.split('+')[0]+'+'+args.mixer)

        subfolder = '{}-paths-reuse_network-{}-full_observable-{}-charging-{}'.format(args.path_number,args.reuse_network,args.full_observable,'-'.join([str(c) for c in args.chargingnodes]))
        self.output_dir = os.path.join(self.output_dir,subfolder)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)          

        print('Init algorithm {} with {}'.format(args.alg,args.mixer))
        
        
    def learn(self, batch, max_episode_len, train_step, epsilon = None,agent_id=None):
        #batch shape: n_episode * max_episode_len * n_agent * state_dim <all agent obs + ID>
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
        
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        
        if self.args.cuda:
            u = u.cuda()
            r= r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()
            s = s.cuda()
            s_next = s_next.cuda()
            
        q_evals = torch.gather(q_evals, dim = 3, index = u).squeeze(3)
        
        q_targets [valid_u_next == 0.0] = -999999
        q_targets = q_targets.max(dim=3)[0]
        #pdb.set_trace()
        if self.args.mixer == 'vdn':
            q_total_evals = self.eval_mixer(q_evals)
            q_total_targets = self.target_mixer(q_targets)
        elif self.args.mixer == 'qmix':
            #hyper net use global state and it does not include agent ID
            hyper_states = s[:,:,0,:-self.n_agents].clone()
            hyper_states_next = s_next[:,:,0,:-self.n_agents].clone()
            q_total_evals, q_total_targets = self.get_q_totals(q_evals, q_targets,\
                            hyper_states, hyper_states_next,max_episode_len)
        else:
            raise ValueError('Mixer {} not recognised'.format(self.args.mixer))
            
        targets = r + self.args.gamma * q_total_targets * (1 - terminated)
        
        td_error =(q_total_evals- targets.detach())
        masked_td_error = mask * td_error
        
        loss = (masked_td_error ** 2).sum()/mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        if train_step >0 and train_step % self.args.target_update_cycle == 0:
            if self.args.reuse_network == True: 
                self.target_rnn.load_state_dict(self.eval_net.state_dict())
            else:
                for agent_id in range(self.n_agents):
                    self.target_rnn[agent_id].load_state_dict(self.eval_net[agent_id].state_dict())                
            self.target_mixer.load_state_dict(self.eval_mixer.state_dict())            
        
    def get_q_values(self,batch,max_episode_len):
        episode_num = batch['s'].shape[0]
        q_evals, q_targets = [], []
        
        if self.args.full_observable == False:
            agent_feature, agent_feature_next = self._get_local_obs(batch, self.args.reuse_network)
        
        for transition_idx in range(max_episode_len):
            if self.args.full_observable:
                if self.args.reuse_network:
                    nn_inputs = batch['s'][:,transition_idx,:]
                    nn_inputs_next = batch['s_next'][:,transition_idx,:]
                else:
                    #exclude agent IDs
                    nn_inputs = batch['s'][:,transition_idx,:,:3 * self.n_agents]
                    nn_inputs_next = batch['s_next'][:,transition_idx,:,:3 * self.n_agents]
            else:                
                nn_inputs = agent_feature[:,transition_idx,:]
                nn_inputs_next = agent_feature_next[:, transition_idx,:]
            #cuda support
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                nn_inputs_next = nn_inputs_next.cuda()
            
            if self.args.reuse_network == True: 
                if self.args.cuda:
                    self.eval_hidden = self.eval_hidden.cuda()
                    self.target_hidden = self.target_hidden.cuda()
                nn_inputs = nn_inputs.reshape(episode_num * self.n_agents, -1)
                nn_inputs_next = nn_inputs_next.reshape(episode_num * self.n_agents, -1)
                q_eval, self.eval_hidden = self.eval_net(nn_inputs, self.eval_hidden)
                q_target, self.target_hidden = self.target_rnn(nn_inputs_next, self.target_hidden)
                
                q_eval = q_eval.reshape(episode_num, self.n_agents,-1)
                q_target = q_target.reshape(episode_num, self.n_agents, -1)
            
            else:
                #shape of nn_inputs: episode_num  * n_agent * feature_len
                q_eval,q_target = [],[]
                for agent_id in range(self.n_agents):
                    if self.args.cuda:
                        self.eval_hidden[agent_id] = self.eval_hidden[agent_id].cuda()
                        self.target_hidden[agent_id] = self.target_hidden[agent_id].cuda()
                    q_eval_agent, self.eval_hidden[agent_id] = self.eval_net[agent_id](nn_inputs[:,agent_id,:],self.eval_hidden[agent_id])
                    q_target_agent, self.target_hidden[agent_id] = self.target_rnn[agent_id](nn_inputs_next[:,agent_id,:],self.target_hidden[agent_id])
                    q_eval.append(q_eval_agent)
                    q_target.append(q_target_agent)
                    
                q_eval = torch.stack(q_eval,dim=1)
                q_target = torch.stack(q_target,dim=1)
                
            q_evals.append(q_eval)
            q_targets.append(q_target)
            
        q_evals = torch.stack(q_evals, dim = 1)
        q_targets = torch.stack(q_targets,dim=1)
                
        return q_evals, q_targets
    
        
    def get_q_totals(self, q_evals, q_targets,hyper_states, hyper_states_next,max_episode_len):
        assert q_evals.shape == q_targets.shape
        #episode_num = q_evals.shape[0]
        q_total_evals, q_total_targets = [],[]
        for transition_idx in range(max_episode_len):
            q_eval = q_evals[:,transition_idx,:]
            q_target = q_targets[:,transition_idx,:]
            nn_inputs = hyper_states[:,transition_idx,:]
            nn_inputs_next = hyper_states_next[:,transition_idx,:]
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                nn_inputs_next = nn_inputs_next.cuda()
                self.eval_mixer_hidden = self.eval_mixer_hidden.cuda()
                self.target_mixer_hidden = self.target_mixer_hidden.cuda()
            q_total_eval, self.eval_mixer_hidden = self.eval_mixer(q_eval, nn_inputs,self.eval_mixer_hidden)
            q_total_target, self.target_mixer_hidden = self.target_mixer(q_target, nn_inputs_next, self.target_mixer_hidden)
            q_total_evals.append(q_total_eval)
            q_total_targets.append(q_total_target)
        q_total_evals = torch.stack(q_total_evals, dim = 1)
        q_total_targets = torch.stack(q_total_targets, dim = 1)
        return q_total_evals,q_total_targets
        
    
    def _get_local_obs(self,batch, reuse_network):
        agent_indices = torch.argmax(batch['s'][:,:,:,-self.args.n_agents:],dim=-1,keepdim=True)
        agent_indices = agent_indices.repeat(1,1,1,3) * 3 + torch.tensor(range(3))
        agent_feature = torch.gather(batch['s'][:,:,:,:3 * self.args.n_agents],dim=-1,index=agent_indices)
        if reuse_network:
            agent_feature = torch.cat((agent_feature, batch['s'][:,:,:,3*self.args.n_agents:]),dim=-1)
    
        agent_indices_next = torch.argmax(batch['s_next'][:,:,:,-self.args.n_agents:],dim=-1,keepdim=True)
        agent_indices_next = agent_indices_next.repeat(1,1,1,3) * 3 + torch.tensor(range(3))
        agent_feature_next = torch.gather(batch['s_next'][:,:,:,:3 * self.args.n_agents],dim=-1,index=agent_indices_next)
        if reuse_network:
            agent_feature_next = torch.cat((agent_feature_next, batch['s_next'][:,:,:,3*self.args.n_agents:]),dim=-1) 
    
        return agent_feature, agent_feature_next
    
    def init_hidden(self,episode_num):
        if self.args.reuse_network == True:    
            self.eval_hidden = self.eval_net.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents,-1).clone()
            self.target_hidden = self.target_rnn.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents, -1).clone()
        else:
            self.eval_hidden,self.target_hidden = [],[]
            for agent_id in range(self.n_agents):
                self.eval_hidden.append(self.eval_net[agent_id].init_hidden().expand(episode_num, -1))
                self.target_hidden.append(self.target_rnn[agent_id].init_hidden().expand(episode_num, -1))
            
        #Hidden state for hypernet is global, not each agent
        if self.args.mixer == 'qmix':
            self.eval_mixer_hidden = self.eval_mixer.init_hidden().expand(episode_num, -1)
            self.target_mixer_hidden = self.target_mixer.init_hidden().expand(episode_num, -1)
            
    def save_model(self,train_step, agent_id = None):
        num = str(train_step // self.args.save_cycle)
        if self.args.reuse_network == True:
            torch.save(self.eval_net.state_dict(), os.path.join(self.output_dir, '{}_rnn.pkl'.format(num)))
        else:
            for agent_id in range(self.n_agents):
                torch.save(self.eval_net[agent_id].state_dict(), os.path.join(self.output_dir, '{}_rnn-agent{}.pkl'.format(num,agent_id)))
        torch.save(self.eval_mixer.state_dict(), os.path.join(self.output_dir, '{}_mixer.pkl'.format(num)))
        
    def load_model(self,folder, num):
        for agent_id in range(self.n_agents):
            rnn_file_name = '{}/{}_rnn-agent{}.pkl'.format(folder,num,agent_id)
            self.eval_net[agent_id].load_state_dict(torch.load(rnn_file_name))
        mixer_file = '{}/{}_mixer.pkl'.format(folder,num)
        self.eval_mixer.load_state_dict(torch.load(mixer_file))
          
        
class ABR_MIX:
    def __init__(self,args):
        #algs works with MARL with team reward
        assert args.alg.startswith('abr') == True, 'alg works with action branching + team reward'
        assert args.n_agents == 1
        assert args.reward_type == 0, 'alg works as MARL with team reward'
        
        self.args = args
        self.n_agents = args.n_agents 
        self.n_robots= args.n_robots
               
        self.eval_net = ABRCell(args.nn_input_dim, args)
        self.target_rnn = ABRCell(args.nn_input_dim, args)

        self.eval_hidden = None
        self.target_hidden = None

        if args.mixer == 'qmix':
            #The global state dim
            self.hyper_input_dim = 3 * self.n_robots
            self.eval_mixer = QMixNet(self.hyper_input_dim, args)
            self.target_mixer = QMixNet(self.hyper_input_dim, args)
            self.eval_mixer_hidden = None
            self.target_mixer_hidden = None
            
        elif args.mixer == 'vdn':
            self.eval_mixer = VDNNet()
            self.target_mixer = VDNNet()
        else:
            raise ValueError('Mixer {} not recognised'.format(args.mixer))
            
        if self.args.cuda:
            
            self.eval_net.cuda()
            self.target_rnn.cuda()

            self.eval_mixer.cuda()
            self.target_mixer.cuda()
        
        self.target_rnn.load_state_dict(self.eval_net.state_dict())
        self.target_mixer.load_state_dict(self.eval_mixer.state_dict())

 
        self.eval_parameters = list(self.eval_mixer.parameters()) + list(self.eval_net.parameters())
        
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr = args.lr)        
        
        self.output_dir = os.path.join(args.output_dir, args.area_name, args.alg.split('+')[0]+'+'+args.mixer)

        subfolder = '{}-paths-charging-{}'.format(args.path_number,'-'.join([str(c) for c in args.chargingnodes]))
        self.output_dir = os.path.join(self.output_dir,subfolder)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)          

        print('Init algorithm {} with {}'.format(args.alg,args.mixer))
        
        
    def learn(self, batch, max_episode_len, train_step, epsilon = None):
        #batch shape: n_episode * max_episode_len * n_agent * state_dim <all agent obs + ID>
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
        
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        
        if self.args.cuda:
            u = u.cuda()
            r= r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()
            s = s.cuda()
            s_next = s_next.cuda()
            
        q_evals = torch.gather(q_evals, dim = 3, index = u).squeeze(3)
        
        q_targets [valid_u_next == 0.0] = -999999
        q_targets = q_targets.max(dim=3)[0]
        #pdb.set_trace()
        if self.args.mixer == 'vdn':
            q_total_evals = self.eval_mixer(q_evals)
            q_total_targets = self.target_mixer(q_targets)
        elif self.args.mixer == 'qmix':
            #hyper net use global state and it does not include agent ID
            hyper_states = s[:,:,0,:].clone()
            hyper_states_next = s_next[:,:,0,:].clone()
            q_total_evals, q_total_targets = self.get_q_totals(q_evals, q_targets,\
                            hyper_states, hyper_states_next,max_episode_len)
        else:
            raise ValueError('Mixer {} not recognised'.format(self.args.mixer))
            
        targets = r + self.args.gamma * q_total_targets * (1 - terminated)
        
        td_error =(q_total_evals- targets.detach())
        masked_td_error = mask * td_error
        
        loss = (masked_td_error ** 2).sum()/mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        if train_step >0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_net.state_dict())
            self.target_mixer.load_state_dict(self.eval_mixer.state_dict())            
        
    def get_q_values(self,batch,max_episode_len):
        episode_num = batch['s'].shape[0]
        q_evals, q_targets = [], []

        for transition_idx in range(max_episode_len):
            nn_inputs = batch['s'][:,transition_idx,:]
            nn_inputs_next = batch['s_next'][:,transition_idx,:]
            #cuda support
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                nn_inputs_next = nn_inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            nn_inputs = nn_inputs.reshape(episode_num * self.n_agents, -1)
            nn_inputs_next = nn_inputs_next.reshape(episode_num * self.n_agents, -1)
            q_eval, self.eval_hidden = self.eval_net(nn_inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(nn_inputs_next, self.target_hidden)
            
            q_eval = torch.stack(q_eval,dim=1)
            q_target = torch.stack(q_target,dim=1)
    
            q_evals.append(q_eval)
            q_targets.append(q_target)
            
        q_evals = torch.stack(q_evals, dim = 1)
        q_targets = torch.stack(q_targets,dim=1)
                
        return q_evals, q_targets
    
        
    def get_q_totals(self, q_evals, q_targets,hyper_states, hyper_states_next,max_episode_len):
        assert q_evals.shape == q_targets.shape
        #episode_num = q_evals.shape[0]
        q_total_evals, q_total_targets = [],[]
        for transition_idx in range(max_episode_len):
            q_eval = q_evals[:,transition_idx,:]
            q_target = q_targets[:,transition_idx,:]
            nn_inputs = hyper_states[:,transition_idx,:]
            nn_inputs_next = hyper_states_next[:,transition_idx,:]
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                nn_inputs_next = nn_inputs_next.cuda()
                self.eval_mixer_hidden = self.eval_mixer_hidden.cuda()
                self.target_mixer_hidden = self.target_mixer_hidden.cuda()
            q_total_eval, self.eval_mixer_hidden = self.eval_mixer(q_eval, nn_inputs,self.eval_mixer_hidden)
            q_total_target, self.target_mixer_hidden = self.target_mixer(q_target, nn_inputs_next, self.target_mixer_hidden)
            q_total_evals.append(q_total_eval)
            q_total_targets.append(q_total_target)
        q_total_evals = torch.stack(q_total_evals, dim = 1)
        q_total_targets = torch.stack(q_total_targets, dim = 1)
        return q_total_evals,q_total_targets
        
    

    
    def init_hidden(self,episode_num):  
        self.eval_hidden = self.eval_net.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents,-1).clone()
        self.target_hidden = self.target_rnn.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents, -1).clone()
        #Hidden state for hypernet is global, not each agent
        if self.args.mixer == 'qmix':
            self.eval_mixer_hidden = self.eval_mixer.init_hidden().expand(episode_num, -1)
            self.target_mixer_hidden = self.target_mixer.init_hidden().expand(episode_num, -1)
            
    def save_model(self,train_step):
        num = str(train_step // self.args.save_cycle)
        torch.save(self.eval_net.state_dict(), os.path.join(self.output_dir, '{}_abr_rnn.pkl'.format(num)))
        torch.save(self.eval_mixer.state_dict(), os.path.join(self.output_dir, '{}_abr_mixer.pkl'.format(num)))
        
                