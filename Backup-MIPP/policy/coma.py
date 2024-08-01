#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:00:24 2020

@author: yongyongwei
"""

import torch
import os
from network.base_net import RNNCell
from network.coma_critic import ComaCritic
from common.utils import td_lambda_target
#import pdb

class COMA:
    def __init__(self,args):
        #algs works with MARL with team reward
        assert args.alg.startswith('ma') == True, 'alg works in multi-agent env'
        assert args.reward_type == 0, 'alg works as MARL with team reward'
        assert args.n_agents > 1, 'n_agents should greater than 1'
    
        self.args = args
        self.n_agents = args.n_agents         
            
        if args.reuse_network == True: 
            self.eval_rnn = RNNCell(args.nn_input_dim, args)
        else:
            self.eval_rnn = []
            for agent_id in range(self.n_agents):
                self.eval_rnn.append(RNNCell(args.nn_input_dim, args))          
        
        self.eval_hidden = None
        
        self.coma_critic_dim = 3 * self.n_agents + self.n_agents
        
        self.eval_critic = ComaCritic(self.coma_critic_dim, args)
        self.target_critic = ComaCritic(self.coma_critic_dim, args)
        self.target_critic.load_state_dict(self.eval_critic.state_dict()) 
        
        if self.args.cuda:
            if args.reuse_network == True: 
                self.eval_rnn.cuda()
            else:
                for agent_id in range(self.n_agents):
                    self.eval_rnn[agent_id].cuda()
                    
            self.eval_critic.cuda()
            self.target_critic.cuda()
        
        if args.reuse_network == True: 
            self.rnn_parameters = list(self.eval_rnn.parameters())
        else:
            self.rnn_parameters = []
            for agent_id in range(self.n_agents):
                self.rnn_parameters.extend(list(self.eval_rnn[agent_id].parameters()))
            
        self.critic_parameters = list(self.eval_critic.parameters())
        
        self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr= args.lr_actor)
        self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
        
        self.output_dir = os.path.join(args.output_dir, args.area_name, args.alg)
        

        subfolder = '{}-paths-reuse_network-{}-full_observable-{}-charging-{}'.format(args.path_number,args.reuse_network,args.full_observable,'-'.join([str(c) for c in args.chargingnodes]))
        self.output_dir = os.path.join(self.output_dir,subfolder)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)         
            


    def learn(self,batch, max_episode_len, train_step, epsilon, agent_id=None):
        episode_num = batch['s'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)   
        u = batch['u']
        mask = (1 - batch['padded'].float()).repeat(1,1,self.n_agents)
        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()
            
        q_values = self._train_critic(batch,max_episode_len, train_step)
        
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)
        
        q_taken = torch.gather(q_values, dim=3, index = u).squeeze(3)
        pi_taken = torch.gather(action_prob, dim = 3, index = u).squeeze(3)
        
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)
        
        baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()
        loss = -((advantage * log_pi_taken) * mask).sum()/mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        #print('training loss',loss.item())
        
    def _train_critic(self, batch, max_episode_len, train_step):
        u, r, valid_u, terminated = batch['u'],batch['r'],batch['valid_u'],batch['terminated']
        u_next = u[:,1:]
        padded_u_next = torch.zeros(*u[:,-1].shape, dtype = torch.long).unsqueeze(1)
        u_next = torch.cat((u_next, padded_u_next),dim=1)
        mask = (1-batch['padded'].float()).repeat(1,1,self.n_agents)
        
        if self.args.cuda:
            u = u.cuda()
            u_next = u_next.cuda()
            mask = mask.cuda()
            
        q_evals, q_next_targets = self._get_q_values(batch, max_episode_len)
        q_values = q_evals.clone()
        
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_next_targets = torch.gather(q_next_targets, dim=3, index=u_next).squeeze(3)
        #q_next_targets = q_next_targets.max(dim=-1)[0]
        targets = td_lambda_target(batch, max_episode_len, q_next_targets.cpu(), self.args)
        if self.args.cuda:
            targets = targets.cuda()
        td_error = targets.detach() - q_evals
        masked_td_error = mask * td_error
        
        loss = (masked_td_error ** 2).sum()/mask.sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        if train_step >0 and train_step % self.args.target_update_cycle==0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
            
        return q_values
        
        
    def _get_q_values(self,batch, max_episode_len): 
        episode_num = batch['s'].shape[0]
        state_dim = batch['s'].shape[-1]
        q_evals, q_targets= [], []
        for transition_idx in range(max_episode_len):
            #inputs: episode_num, (transitiondix+1), n_agent, state_shape
            #inputs_next: episode_num, (transitiondix+2), n_agent, state_shape
            inputs,inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            inputs = inputs.permute(0,2,1,3).reshape(episode_num * self.n_agents,-1, state_dim)
            inputs_next = inputs_next.permute(0,2,1,3).reshape(episode_num * self.n_agents,-1, state_dim)
            q_eval = self.eval_critic(inputs)
            q_target = self.target_critic(inputs_next)
            q_eval = q_eval.view(episode_num, self.n_agents,-1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets,dim=1)    
        return q_evals, q_targets
    
    def _get_critic_inputs(self,batch, transition_idx, max_episode_len):
        #alter the states and assume the agent does not take any movement
        if transition_idx == 0:
            inputs = batch['s'][:,transition_idx,:,:].unsqueeze(1)
        else:
            #exclude agent ID first, later will concatenate
            s_pre = batch['s'][:,:transition_idx,:,:-self.n_agents]
            #mask 1, used to retain info of one agent from the previous step 
            #also note 3 is the observation size for each robot (x,y,remaining_budget)
            mask1 = torch.eye(self.n_agents).reshape(-1,1).repeat(1,3).view(self.n_agents,-1).unsqueeze(0)
            old_state=batch['s'][:,transition_idx-1,:,:-self.n_agents] * mask1
            #mask2 used to keep the info of other agents from the current step
            mask2 = (1-torch.eye(self.n_agents)).reshape(-1,1).repeat(1,3).view(self.n_agents,-1).unsqueeze(0)
            other_agent_states = batch['s'][:,transition_idx,:,:-self.n_agents] * mask2
            merged = (old_state + other_agent_states).unsqueeze(1)
            inputs = torch.cat((s_pre,merged),dim=1)
            #concat agent ID
            inputs = torch.cat((inputs,batch['s'][:,:transition_idx+1,:,-self.n_agents:]),dim=-1)
            
        #Do the same for inputs_next
        s_pre = batch['s'][:,:transition_idx+1,:,:-self.n_agents]
        mask1 = torch.eye(self.n_agents).reshape(-1,1).repeat(1,3).view(self.n_agents,-1).unsqueeze(0)
        old_state = batch['s'][:,transition_idx,:,:-self.n_agents] * mask1        
        mask2 = (1-torch.eye(self.n_agents)).reshape(-1,1).repeat(1,3).view(self.n_agents,-1).unsqueeze(0)
        other_agent_states = batch['s_next'][:,transition_idx,:,:-self.n_agents] * mask2
        merged = (old_state + other_agent_states).unsqueeze(1)
        inputs_next = torch.cat((s_pre,merged),dim=1)
        agentID = batch['s'][:,:transition_idx+1,:,-self.n_agents:]
        #add the agentID for the new transition
        agentID = torch.cat((agentID,batch['s_next'][:,transition_idx,:,-self.n_agents:].unsqueeze(1)),dim=1)
        inputs_next = torch.cat((inputs_next,agentID),dim=-1)
        
        #inputs_next is 1 step longer than inputs, since RNN is not cell but rolled seq
        return inputs, inputs_next
            
            

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        #here valid actions is one-hot encoded instead of action number
        episode_num = batch['s'].shape[0]
        valid_actions = batch['valid_u']
        action_logits = []
        #prepare for partial observable case
        if self.args.full_observable == False:
            agent_indices = torch.argmax(batch['s'][:,:,:,-self.args.n_agents:],dim=-1,keepdim=True)
            agent_indices = agent_indices.repeat(1,1,1,3) * 3 + torch.tensor(range(3))
            agent_feature = torch.gather(batch['s'][:,:,:,:3 * self.args.n_agents],dim=-1,index=agent_indices)
            if self.args.reuse_network:
                agent_feature = torch.cat((agent_feature, batch['s'][:,:,:,3*self.args.n_agents:]),dim=-1)
        #loop for each transition
        for transition_idx in range(max_episode_len):
            if self.args.full_observable:
                if self.args.reuse_network:
                    nn_inputs = batch['s'][:,transition_idx,:]
                else:
                    nn_inputs = batch['s'][:,transition_idx,:,:3 * self.n_agents]    
            else:
                nn_inputs = agent_feature[:,transition_idx,:]
                
            if self.args.cuda:
                nn_inputs = nn_inputs.cuda()
                
            if self.args.reuse_network == True: 
                if self.args.cuda:
                    self.eval_hidden = self.eval_hidden.cuda()   
                nn_inputs = nn_inputs.reshape(episode_num * self.n_agents, -1)
                #the forward function of rnn will reshape eval_hidden
                outputs, self.eval_hidden = self.eval_rnn(nn_inputs, self.eval_hidden)
                outputs = outputs.reshape(episode_num, self.n_agents, -1)
                action_logits.append(outputs)
            else:
                outputs = []
                for agent_id in range(self.n_agents):
                    if self.args.cuda:
                        self.eval_hidden[agent_id] = self.eval_hidden[agent_id].cuda()
                    output, self.eval_hidden[agent_id] = self.eval_rnn[agent_id](nn_inputs[:,agent_id,:],self.eval_hidden[agent_id])
                    outputs.append(output)
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
        if self.args.reuse_network == True: 
            self.eval_hidden = self.eval_rnn.init_hidden().unsqueeze(0).expand(episode_num, self.n_agents,-1)
        else:
            self.eval_hidden = []
            for agent_id in range(self.n_agents):
                self.eval_hidden.append(self.eval_rnn[agent_id].init_hidden().expand(episode_num, -1))
            
    def save_model(self,train_step,agent_id = None):
        num = str(train_step // self.args.save_cycle)
        if self.args.reuse_network == True:
            torch.save(self.eval_rnn.state_dict(), os.path.join(self.output_dir, '{}_rnn.pkl'.format(num)))
        else:
            for agent_id in range(self.n_agents):
                torch.save(self.eval_rnn[agent_id].state_dict(), os.path.join(self.output_dir, '{}_rnn-agent{}.pkl'.format(num,agent_id)))
        torch.save(self.eval_critic.state_dict(), os.path.join(self.output_dir, '{}_critic.pkl'.format(num)))
        
                    