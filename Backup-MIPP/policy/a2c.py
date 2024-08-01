#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:48:32 2020

@author: yongyongwei
"""

import torch 
import os
from network.base_net import RNNCell,ABRCell, MLP

class A2C:
    def __init__(self,args):
        
        #force n_agents = 1, and runner will create multi-agent instances
        if args.n_agents > 1 and args.reuse_network == False:
            self.n_agents = 1
        else:
            self.n_agents = args.n_agents        
        
        self.args = args
        
        if args.state_type ==0:
            self.eval_net = RNNCell(args.nn_input_dim, args)
            self.eval_critic = RNNCell(args.critic_dim, args, critic = True)
            self.target_critic = RNNCell(args.critic_dim, args, critic = True)
        else:
            self.eval_net = MLP(args.nn_input_dim, args)
            self.eval_critic = MLP(args.critic_dim, args, critic = True)
            self.target_critic = MLP(args.critic_dim, args, critic = True)            
            
        if self.args.cuda:
            self.eval_net.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()
            
            
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.nn_parameters = list(self.eval_net.parameters())
        self.critic_parameters = list(self.eval_critic.parameters()) 

        self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr = args.lr_critic)
        self.nn_optimizer = torch.optim.RMSprop(self.nn_parameters, lr = args.lr_actor)
        
        self.eval_hidden = None
        self.eval_critic_hidden = None
        self.target_critic_hidden = None
               
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
            
            

    def learn(self, batch, max_episode_len, train_step, epsilon,agent_id=None):
        if self.args.state_type ==0:  
            episode_num = batch['s'].shape[0]
            self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)
        #Expand reward signals if nessesary
        if self.n_agents > batch['r'].shape[-1]:
            if self.args.state_type ==0:
                batch['r'] = batch['r'].expand(-1,-1,self.n_agents)
            else:
                batch['r'] = batch['r'].expand(-1,self.n_agents)

        u, r, valid_u, terminated = batch['u'], batch['r'], batch['valid_u'],batch['terminated']    
        
        if self.args.state_type==0: 
            mask = (1 - batch['padded'].float()).repeat(1,1,self.n_agents)
        else:
            mask = None
        if self.args.cuda:
            r = r.cuda()
            u = u.cuda()
            terminated = terminated.cuda()    
            if self.args.state_type==0:
                mask = mask.cuda()                
        
        td_error = self._train_critic(batch, max_episode_len, train_step)
        
        action_prob = self._get_action_prob(batch,max_episode_len, epsilon,agent_id)
        
        pi_taken = torch.gather(action_prob, dim = -1, index = u).squeeze(-1)
        
        if self.args.state_type ==0:
            pi_taken[mask ==0] = 1.0
            log_pi_taken = torch.log(pi_taken)
            loss =  - ((td_error.detach() * log_pi_taken) * mask).sum()/mask.sum()
            
        else:
            log_pi_taken = torch.log(pi_taken)
            loss =  - ((td_error.detach() * log_pi_taken)).mean()
        
        self.nn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn_parameters, self.args.grad_norm_clip)
        self.nn_optimizer.step()
        #print('Actor loss is ',loss)

        
    def _train_critic(self,batch, max_episode_len, train_step):
        #critic always has full observibility
        r, terminated = batch['r'], batch['terminated']
        if self.args.state_type==0:
            mask = 1 - batch['padded'].float()
        else:
            mask = None
        if self.args.cuda:
            r= r.cuda()
            terminated = terminated.cuda()
            if self.args.state_type ==0:
                mask = mask.cuda()
            
        v_evals, v_next_targets = self._get_v_values(batch, max_episode_len)
        
        v_targets = self._get_nstep_targets(r, v_next_targets, max_episode_len, mask, terminated)
        td_error = v_targets.detach() - v_evals
        
        if self.args.state_type==0:
            mask = mask.repeat(1,1,self.n_agents)
            masked_td_error = mask * td_error
            
            loss = (masked_td_error ** 2).sum()/mask.sum()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
            self.critic_optimizer.step()
        else:
            loss = (td_error **2).mean()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()           
        
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
            
        return td_error
                
    def _get_nstep_targets(self, r, v_next_targets, max_episode_len, mask, terminated):
        if self.args.nstep_return == 1:
            return r + self.args.gamma * v_next_targets * (1 - terminated)
        elif self.args.state_type ==0:
            episode_num = r.shape[0]
            nstep_targets = torch.zeros_like(r)
            for transition_idx in range(max_episode_len):
                i = transition_idx + self.args.nstep_return
                if i>=max_episode_len:
                    r_nstep = r[:,transition_idx: ,:] * mask[:,transition_idx:,:]
                    discounter = torch.tensor([self.args.gamma ** j for j in range(r_nstep.shape[1])]).reshape(1,-1,1).expand(episode_num,-1,self.n_agents)
                    nstep_targets[:,transition_idx,:] = torch.sum(r_nstep * discounter, dim = 1)
                else:
                    r_nstep = r[:,transition_idx:i ,:] * mask[:,transition_idx:i,:]
                    future_v = v_next_targets[:,i:(i+1),:] * (1 - terminated)[:,i:(i+1),:]
                    r_nstep = torch.cat((r_nstep,future_v),dim=1) 
                    discounter = torch.tensor([self.args.gamma ** j for j in range(self.args.nstep_return+1)]).reshape(1,-1,1).expand(episode_num,-1,self.n_agents)
                    nstep_targets[:,transition_idx,:] = torch.sum(r_nstep * discounter, dim = 1)
                
            return nstep_targets
        else:
            raise Exception("Not implemented yet!")
                    
                    
                
                
            
            
    def _get_v_values(self,batch, max_episode_len):
        if self.args.state_type ==0:
            episode_num = batch['s'].shape[0]
            v_evals, v_next_targets = [], []
            for transition_idx in range(max_episode_len):
                nn_inputs = batch['s'][:,transition_idx].reshape(episode_num * self.n_agents, -1)
                nn_inputs_next = batch['s_next'][:,transition_idx].reshape(episode_num * self.n_agents, -1)
                if self.args.cuda:
                    nn_inputs = nn_inputs.cuda()
                    nn_inputs_next = nn_inputs_next.cuda()
                    self.eval_critic_hidden = self.eval_critic_hidden.cuda()
                    self.target_critic_hidden = self.target_critic_hidden.cuda()
                v_eval, self.eval_critic_hidden = self.eval_critic(nn_inputs, self.eval_critic_hidden)
                v_next_target, self.target_critic_hidden = self.target_critic(nn_inputs_next, self.target_critic_hidden)
                
                v_eval = v_eval.reshape(episode_num, self.n_agents,-1)
                v_next_target = v_next_target.reshape(episode_num, self.n_agents, -1)
                v_evals.append(v_eval)
                v_next_targets.append(v_next_target)
            v_evals = torch.stack(v_evals, dim = 1)
            v_next_targets = torch.stack(v_next_targets,dim=1)
            return v_evals.squeeze(-1), v_next_targets.squeeze(-1)  
        else:
            batch_size = batch['s'].shape[0]
            nn_inputs = torch.cat((batch['nodes_info'],batch['s']),dim=-1).reshape(batch_size * self.n_agents, -1)
            nn_inputs_next = torch.cat((batch['nodes_info_next'],batch['s_next']),dim=-1).reshape(batch_size * self.n_agents, -1)
            v_eval = self.eval_critic(nn_inputs)
            v_next_target = self.target_critic(nn_inputs_next)
            v_eval = v_eval.reshape(batch_size, self.n_agents, -1)
            v_next_target = v_next_target.reshape(batch_size, self.n_agents,-1)
            return v_eval.squeeze(-1), v_next_target.squeeze(-1) 
            
    def _get_action_prob(self, batch, max_episode_len, epsilon,agent_id):
        if self.args.state_type ==0:
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
                    # (partial obseve, multi-agents)
                    if self.args.reuse_network:
                        #in this case, agent_id already one hot encoded at the last
                        #also, here self.n_agents = self.args.n_agents > 1
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
                outputs, self.eval_hidden = self.eval_net(nn_inputs, self.eval_hidden)
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
                
        else:
            batch_size = batch['s'].shape[0]
            valid_actions = batch['valid_u']
            action_logits = []
            assert self.args.full_observable == True
            
            nn_inputs = torch.cat((batch['nodes_info'],batch['s']),dim=-1).reshape(batch_size * self.n_agents, -1)
            action_logits = self.eval_net(nn_inputs)
            action_logits = action_logits.reshape(batch_size, self.n_agents, -1)
            
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
        

    def init_hidden(self,episode_num):
        self.eval_hidden = self.eval_net.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents, -1)
        self.eval_critic_hidden = self.eval_critic.init_hidden().unsqueeze(0).expand(episode_num,self.n_agents,-1)
        self.target_critic_hidden = self.target_critic.init_hidden().unsqueeze(0).expand(episode_num,self.n_agents,-1)
        
    def save_model(self,train_step,agent_id=None):
        num = str(train_step // self.args.save_cycle)
        if self.args.alg.startswith('ma'):
            if agent_id is None:
                #if reuse_network True
                model_name = '{}_nn_rt{}_st{}.pkl'.format(num,self.args.reward_type,self.args.state_type)
            else:
                model_name = '{}_nn_rt{}_st{}-agent{}.pkl'.format(num,self.args.reward_type,self.args.state_type,agent_id)
        else:
            model_name = '{}_nn_st{}.pkl'.format(num, self.args.state_type)
        torch.save(self.eval_net.state_dict(), os.path.join(self.output_dir, model_name))
        
        
class ABR_A2C:
    def __init__(self,args):
        self.n_agents = args.n_agents      
        assert self.n_agents == 1
        print('init policy')
        self.args = args
        
        if args.state_type ==0:
            
            self.eval_net = ABRCell(args.nn_input_dim, args)
            self.eval_critic = ABRCell(args.critic_dim, args, critic = True)
            self.target_critic = ABRCell(args.critic_dim, args, critic = True)
        else:
            self.eval_net = MLP(args.nn_input_dim, args)
            self.eval_critic = MLP(args.critic_dim, args, critic = True)
            self.target_critic = MLP(args.critic_dim, args, critic = True)                 
        
        if self.args.cuda:
            self.eval_net.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()
        
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.nn_parameters = list(self.eval_net.parameters())
        self.critic_parameters = list(self.eval_critic.parameters()) 

        self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr = args.lr_critic)
        self.nn_optimizer = torch.optim.RMSprop(self.nn_parameters, lr = args.lr_actor)
        
        self.eval_hidden = None
        self.eval_critic_hidden = None
        self.target_critic_hidden = None
               
        self.output_dir = os.path.join(args.output_dir, args.area_name, args.alg)

        subfolder = '{}-paths-charging-{}'.format(args.path_number,'-'.join([str(c) for c in args.chargingnodes]))
        self.output_dir = os.path.join(self.output_dir,subfolder)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)        
            
    def learn(self, batch, max_episode_len, train_step, epsilon):
        pass
        episode_num = batch['s'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)
        #Expand reward signals if nessesary
        if self.args.n_robots > batch['r'].shape[-1]:
            batch['r'] = batch['r'].expand(-1,-1,self.args.n_robots)

        u, r, valid_u, terminated = batch['u'], batch['r'], batch['valid_u'],batch['terminated']    
        mask = (1 - batch['padded'].float()).repeat(1,1,self.args.n_robots)
        if self.args.cuda:
            r = r.cuda()
            u = u.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()                    
        
        td_error = self._train_critic(batch, max_episode_len, train_step)
        
        action_prob = self._get_action_prob(batch,max_episode_len, epsilon)
        
        pi_taken = torch.gather(action_prob, dim = 3, index = u).squeeze(3)
        pi_taken[mask ==0] = 1.0
        log_pi_taken = torch.log(pi_taken)
        
        loss =  - ((td_error.detach() * log_pi_taken) * mask).sum()/mask.sum()
        
        self.nn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn_parameters, self.args.grad_norm_clip)
        self.nn_optimizer.step()
        #print('Actor loss is ',loss)

        
    def _train_critic(self,batch, max_episode_len, train_step):
        #critic always has full observibility
        r, terminated = batch['r'], batch['terminated']
        mask = 1 - batch['padded'].float()
        if self.args.cuda:
            mask = mask.cuda()
            r= r.cuda()
            terminated = terminated.cuda()
            
        v_evals, v_next_targets = self._get_v_values(batch, max_episode_len)
        
        v_targets = self._get_nstep_targets(r, v_next_targets, max_episode_len, mask, terminated)
        td_error = v_targets.detach() - v_evals
        mask = mask.repeat(1,1,self.n_agents)
        masked_td_error = mask * td_error
        
        loss = (masked_td_error ** 2).sum()/mask.sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
            
        return td_error
                
    def _get_nstep_targets(self, r, v_next_targets, max_episode_len, mask, terminated):
        if self.args.nstep_return == 1:
            return r + self.args.gamma * v_next_targets * (1 - terminated)
        else:
            episode_num = r.shape[0]
            nstep_targets = torch.zeros_like(r)
            for transition_idx in range(max_episode_len):
                i = transition_idx + self.args.nstep_return
                if i>=max_episode_len:
                    r_nstep = r[:,transition_idx: ,:] * mask[:,transition_idx:,:]
                    discounter = torch.tensor([self.args.gamma ** j for j in range(r_nstep.shape[1])]).reshape(1,-1,1).expand(episode_num,-1,self.n_agents)
                    nstep_targets[:,transition_idx,:] = torch.sum(r_nstep * discounter, dim = 1)
                else:
                    r_nstep = r[:,transition_idx:i ,:] * mask[:,transition_idx:i,:]
                    future_v = v_next_targets[:,i:(i+1),:] * (1 - terminated)[:,i:(i+1),:]
                    r_nstep = torch.cat((r_nstep,future_v),dim=1) 
                    discounter = torch.tensor([self.args.gamma ** j for j in range(self.args.nstep_return+1)]).reshape(1,-1,1).expand(episode_num,-1,self.n_agents)
                    nstep_targets[:,transition_idx,:] = torch.sum(r_nstep * discounter, dim = 1)
                
            return nstep_targets
            
    def _get_v_values(self,batch, max_episode_len):
        episode_num = batch['s'].shape[0]
        v_evals, v_next_targets = [], []
        for transition_idx in range(max_episode_len):
            nn_inputs = batch['s'][:,transition_idx].reshape(episode_num * self.n_agents, -1)
            nn_inputs_next = batch['s_next'][:,transition_idx].reshape(episode_num * self.n_agents, -1)
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                nn_inputs_next = nn_inputs_next.cuda()
                self.eval_critic_hidden = self.eval_critic_hidden.cuda()
                self.target_critic_hidden = self.target_critic_hidden.cuda()
            v_eval, self.eval_critic_hidden = self.eval_critic(nn_inputs, self.eval_critic_hidden)
            v_next_target, self.target_critic_hidden = self.target_critic(nn_inputs_next, self.target_critic_hidden)
            
            v_eval = torch.stack(v_eval,dim=1)
            v_next_target = torch.stack(v_next_target,dim=1)

            v_evals.append(v_eval)
            v_next_targets.append(v_next_target)
            
        v_evals = torch.stack(v_evals, dim = 1)
        v_next_targets = torch.stack(v_next_targets,dim=1)
        
        return v_evals.squeeze(-1), v_next_targets.squeeze(-1)                
            
    def _get_action_prob(self, batch, max_episode_len, epsilon):
        #here valid actions is one-hot encoded instead of action number
        episode_num = batch['s'].shape[0]
        valid_actions = batch['valid_u']
        action_logits = []
        for transition_idx in range(max_episode_len):
            nn_inputs = batch['s'][:,transition_idx,:].reshape(episode_num * self.n_agents, -1)
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            #the forward function of rnn will reshape eval_hidden
            outputs, self.eval_hidden = self.eval_net(nn_inputs, self.eval_hidden)
            outputs = torch.stack(outputs,dim=1)
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
        

    def init_hidden(self,episode_num):
        self.eval_hidden = self.eval_net.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents, -1)
        self.eval_critic_hidden = self.eval_critic.init_hidden().unsqueeze(0).expand(episode_num,self.n_agents,-1)
        self.target_critic_hidden = self.target_critic.init_hidden().unsqueeze(0).expand(episode_num,self.n_agents,-1)
        
    def save_model(self,train_step):
        num = str(train_step // self.args.save_cycle)
        model_name = '{}_abr_nn_rt{}_st{}.pkl'.format(num,self.args.reward_type,self.args.state_type)
        torch.save(self.eval_net.state_dict(), os.path.join(self.output_dir, model_name))
        

          