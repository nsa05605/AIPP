#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:02:42 2020

Learn I-G-M equation:
    
    In this case, the individual Q-function is opitimistic projection (best match)
@author: yongyongwei
"""

import torch
import os,sys
from network.base_net import RNNCell
from network.base_net import MLP
import copy,itertools
import numpy as np
from torch.nn.functional import one_hot

class TESTJAL:
    """
    This is simply used to validate the Joint Q learned from optimistic Q
    """
    def __init__(self,args):
        #force n_agents = 1, and runner will create multi-agent instances
        self.n_agents = args.n_agents
        #assert self.n_agents == 1
        
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
            
        #assert self.n_agents == 1
        
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
            
        return q_evals.clone().detach()
        
        
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
        

def _get_joint_action_indices(valid_actions):
    action_size = 28
    indices=[]
    for joint_action in itertools.product(*valid_actions):
        index=0
        for i,action_num in enumerate(joint_action):
            index += (action_size ** i) * action_num
        #note convert to integer since it is index
        indices.append(int(index))
    return indices

def _get_joint_action_from_index(action_index):
    #only consier the 2 robot case
    actions=[]
    action_size = 28
    if action_index<action_size:
        actions.append(action_index)
    else:
        while(action_index//action_size >=1):
            actions.append(action_index % action_size)
            action_index = action_index //action_size
        if (action_index % action_size !=0):
            actions.append(action_index % action_size)
    if len(actions) < 2:
        actions =  actions + [0] * (2 - len(actions))
    return actions
    
def convertBatchtoJAL(batch_in):
    """convert the batch format to JAL batch format"""
    batch = copy.deepcopy(batch_in)
    b_size, epi_len, n_agent, state_size = batch_in['s'].shape
    batch['s'] = batch['s'][:,:,0:1,:3 * n_agent]
    batch['s_next'] = batch['s_next'][:,:,0:1,:3 * n_agent]
    
    u_index = np.zeros((b_size,epi_len,1))
    for i in range(b_size):
        for j in range(epi_len):
            u_index[i,j,0] = _get_joint_action_indices([[batch['u'][i,j,0,0]],[batch['u'][i,j,1,0]]])[0]

    #u_index = torch.tensor(u_index,dtype = torch.long)

    batch['u_index'] = u_index
    #batch['u_onehot'] = one_hot(u_index, 784)
    #the one_hot function needs inputs a tensor
    batch['u_onehot'] = one_hot(torch.tensor(u_index,dtype = torch.long), 784).numpy()
    
    batch['r'] = batch['r'].sum(axis=-1,keepdims=True) 
    
    valid_u = np.zeros((b_size,epi_len,1,784))
    valid_u_next = np.zeros((b_size,epi_len,1,784))
    
    for i in range(b_size):
        for j in range(epi_len):
            actions_1 = np.where(batch['valid_u'][i,j,0,:]==1)[0].tolist()
            actions_2 = np.where(batch['valid_u'][i,j,1,:]==1)[0].tolist()
            valid_actions = _get_joint_action_indices([actions_1,actions_2])
            one_hot_valid = np.zeros(784)
            one_hot_valid[valid_actions] = 1
            valid_u[i,j,0,:] = one_hot_valid
            
            actions_1 = np.where(batch['valid_u_next'][i,j,0,:]==1)[0].tolist()
            actions_2 = np.where(batch['valid_u_next'][i,j,1,:]==1)[0].tolist()
            valid_actions_next = _get_joint_action_indices([actions_1,actions_2])
            one_hot_valid_next = np.zeros(784)
            one_hot_valid_next[valid_actions_next] = 1
            valid_u_next[i,j,0,:] = one_hot_valid_next
    batch['valid_u'] = valid_u
    batch['valid_u_next'] = valid_u_next
    
    return batch
    

#==========================Start of LIGM=======================================
class LIGM:
    """
    Learning through IGM Loss function
    """
    def __init__(self, args):
        #algs works with MARL with team reward
        assert args.alg.startswith('ma') == True, 'alg works as MARL with team reward'
        assert args.n_agents > 1
        assert args.n_agents == args.n_robots
        #note that in the data structure, it is equally split among active members, 
        #need to sum to get team reward
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
        
        #set cricit = True, so output a scalar.
        self.eval_joint_q = RNNCell(args.nn_input_dim + args.n_actions * args.n_agents, args, critic=True)
        
        #validate with JAL if nessesary
        if self.args.ligm_validate_with_jal:
            if args.n_robots >=3:
                self.args.ligm_validate_with_jal = False
        if self.args.ligm_validate_with_jal:
            args2 = copy.copy(args)
            args2.n_agents = 1
            args2.n_robots = 2
            args2.n_actions = 784
            args2.single_robot_action_size = 28
            self.jalpolicy = TESTJAL(args2)
        else:
            self.jalpolicy = None
        
        self.eval_hidden = None
        self.target_hidden = None
        
        #hidden state for the joint Q-function
        self.eval_hidden_joint = None
        #self.target_hidden_joint = None #not needed

        for agent_id in range(self.n_agents):
            self.target_rnn[agent_id].load_state_dict(self.eval_net[agent_id].state_dict())
        #self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())
        
        eval_net_parameters = []
        for agent_id in range(self.n_agents):
            eval_net_parameters.extend(list(self.eval_net[agent_id].parameters()))
        self.individual_Q_parameters = eval_net_parameters 
        #We use two optimizer for individual Q and joint Q separately
        self.optimizer_individual_Q = torch.optim.RMSprop(self.individual_Q_parameters, lr = args.lr) 
        self.optimizer_joint_Q = torch.optim.RMSprop(self.eval_joint_q.parameters(), lr = args.lr)
        
        #an alternative optimization method
        self.params_all = self.individual_Q_parameters + list(self.eval_joint_q.parameters())
        self.optimizer_all = torch.optim.RMSprop(self.params_all , lr = args.lr)
       
        self.output_dir = os.path.join(args.output_dir, args.area_name, args.alg)
        subfolder = '{}-paths-reuse_network-{}-full_observable-{}-charging-{}'.format(args.path_number,args.reuse_network,args.full_observable,'-'.join([str(c) for c in args.chargingnodes]))
        self.output_dir = os.path.join(self.output_dir,subfolder)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)          

        print('Init algorithm {}'.format(args.alg))
        
        self.grad_norm_all=[]
        self.grad_norm_jq=[]
        self.grad_norm_indq=[]
        
        #self.grad_norm_truejq = []
        self.true_jq_evals = []
        self.ligm_jq_evals = []
        self.best_response_q = []
        self.batch_masks = []
        
        #track the loss
        self.loss_ind_q = []
        self.loss_joint_q = []
        self.loss_igm = []
        

        
    def learn(self, batch, max_episode_len, train_step, epsilon = None,agent_id=None): 
        if self.args.ligm_validate_with_jal:
            batchjal = convertBatchtoJAL(batch)
            true_jq_eval = self.jalpolicy.learn(batchjal,max_episode_len,train_step)
            true_jq_eval = true_jq_eval.squeeze(-1)
            self.true_jq_evals.append(true_jq_eval)
      
        episode_num = batch['s'].shape[0]
        #init hidden for both individual Q and joint Q
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key],dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key],dtype = torch.float32)   
        u, r, valid_u, valid_u_next, terminated = batch['u'],\
            batch['r'], batch['valid_u'],batch['valid_u_next'],batch['terminated']
    
        mask = (1 - batch['padded'].float()).squeeze(-1)
        #save for later statistic purpose
        self.batch_masks.append(mask.clone().detach())
        
        #get individual Q
        ind_q_evals, ind_q_targets, hidden_evals, hidden_targets = self._get_individual_q(batch, max_episode_len)
        ind_q_evals[valid_u == 0 ] = -999999
        ind_q_targets[valid_u_next == 0] = -999999

        #get eval joint Q
        joint_q_evals = self._get_joint_q(batch,max_episode_len).squeeze(-1)
        #save for statistic
        self.ligm_jq_evals.append(joint_q_evals.clone().detach())
        
        #aggregate for individual best-response Q, 0: max of max, 1:mean of max
        if self.args.ligm_agg_method == 0:
            #used for IGM loss as the target (subtract it)
            ind_q_evals_max = ind_q_evals.max(-1)[0].max(-1)[0].clone().detach()
            ind_q_targets_max = ind_q_targets.max(-1)[0].max(-1)[0].clone().detach()
        elif self.args.ligm_agg_method == 1:
            ind_q_evals_max = ind_q_evals.max(-1)[0].mean(-1).clone().detach()
            ind_q_targets_max = ind_q_targets.max(-1)[0].mean(-1).clone().detach()
        else:
            print('illegal aggregation method')
            sys.exit(-1)
                    
        #get the team reward, not in the buffer the reward is splitted among active robots
        r = batch['r'].sum(-1)
        #1. First optimize the joint q function
        y_dqn_joint = r.clone() + self.args.gamma * ind_q_targets_max * (1 - terminated.squeeze(-1))
        td_error_joint = joint_q_evals - y_dqn_joint.detach()
        loss_jq = ((td_error_joint * mask) **2).sum()/mask.sum()
        
    
        #optimize joint Q, this can be integerated 
        """
        self.optimizer_joint_Q.zero_grad()
        loss_jq.backward()
        grad_norm_jq = torch.nn.utils.clip_grad_norm_(self.eval_joint_q.parameters(), self.args.grad_norm_clip)
        self.optimizer_joint_Q.step()
        self.grad_norm_jq.append(grad_norm_jq)
        """
        
        
        #2. Second optimize the individual optimistics Q-function + IGM loss based on the joint Q values
        #a. the td loss for optimistic projection
        #need to extend the reward to every agent
        action_q_value = torch.gather(ind_q_evals, dim=-1, index = u).squeeze(-1)
        #just save for later analysis
        self.best_response_q.append(action_q_value.clone().detach())
        y_dqn_ind = r.unsqueeze(-1).repeat(1,1,self.n_agents) + self.args.gamma * ind_q_targets.max(-1)[0] * (1 - terminated.expand(-1,-1,self.n_agents))
    
        #=============orignal TD error==============================
        td_error_ind = y_dqn_ind - action_q_value
        
        #optimistic learning through adjusting negative td-error
        if self.args.ligm_opt_method == 0: # use a discount for negative weight
            neg_weight = max( 1 - train_step * (1 - self.args.ligm_opt_fix_weight)/self.args.ligm_opt_decaystep, self.args.ligm_opt_fix_weight)
            weights_matrix = (td_error_ind >= 0) * (1.)
            weights_matrix = torch.max(weights_matrix, torch.zeros_like(weights_matrix) + neg_weight)
            td_error_ind = td_error_ind * weights_matrix
        elif self.args.ligm_opt_method == 1: #use a threshold
            neg_thres = max( self.args.ligm_opt_init_thresh - train_step * (self.args.ligm_opt_init_thresh - self.args.ligm_opt_fix_thresh)/self.args.ligm_opt_decaystep, self.args.ligm_opt_fix_thresh)
            td_error_ind = torch.max(td_error_ind, torch.zeros_like(td_error_ind) - neg_thres)
        
        
        #here is multi-agent, needs to expand the mask
        mask_expanded = mask.unsqueeze(-1).expand(-1,-1,self.n_agents)
        loss_ind = ((td_error_ind * mask_expanded) ** 2).sum()/mask_expanded.sum()
        
        #b. the IGM loss, get the soft-argmax, then get the corresponding joint-Q
        soft_actions = []
        for transition_idx in range(max_episode_len):
            q_eval = ind_q_evals[:,transition_idx,:,:]
            soft_actions.append(torch.softmax(q_eval * self.args.ligm_temp, dim = -1).reshape(episode_num,-1))
            #soft_actions.append(q_eval.reshape(episode_num,-1))
        soft_actions = torch.stack(soft_actions,dim=1)
        igm_eval = self._get_joint_q(batch,max_episode_len, soft_actions)
        igm_error = igm_eval.squeeze(-1) - ind_q_evals_max#.detach()
        igm_loss = ((igm_error * mask) ** 2).sum()/mask.sum()
        
        
        #track the loss
        self.loss_ind_q.append(loss_ind.item())
        self.loss_joint_q.append(loss_jq.item())
        self.loss_igm.append(igm_loss.item())
        
        
        """
        loss_total = loss_ind + self.args.ligm_lambda * igm_loss
        self.optimizer_individual_Q.zero_grad()
        loss_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.individual_Q_parameters, self.args.grad_norm_clip)
        self.optimizer_individual_Q.step()
        self.grad_norm_indq.append(grad_norm)
        """
        

        
        #optimize the parameters as a whole
        
        loss_all = loss_ind + self.args.ligm_lambda * igm_loss + self.args.ligm_beta *loss_jq
        self.optimizer_all.zero_grad()
        loss_all.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params_all, self.args.grad_norm_clip)
        self.grad_norm_all.append(grad_norm)
        self.optimizer_all.step()     
        
        
        
        if train_step >0 and train_step % self.args.target_update_cycle == 0:
            for agent_id in range(self.n_agents):
                self.target_rnn[agent_id].load_state_dict(self.eval_net[agent_id].state_dict())  
                #print('negative td error weights:%f' % (neg_weight))
                              
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
    
    def _get_joint_q(self,batch,max_episode_len, soft_actions= None):
        """note: only get the joint_q for the current step, the target is approximated
                 through the individual Q target taking max.
                 soft_actions shape (batch_size, max_episode_len, action_size * n_agents)
        """
        episode_num = batch['s'].shape[0]
        joint_q_evals = []
        if soft_actions is not None:
            soft_actions_hidden = self.eval_joint_q.init_hidden().expand(episode_num, -1)
        for transition_idx in range(max_episode_len):
            #exclude agent IDs, also note only took the obs from 1st agent (since every agent has global obs)
            nn_inputs = batch['s'][:,transition_idx, 0,:3 * self.n_agents]
            if soft_actions is None:
                nn_inputs = torch.cat([nn_inputs,batch['u_onehot'][:,transition_idx,:,:].reshape(episode_num, -1)],dim=-1)
                jq_eval,self.eval_hidden_joint= self.eval_joint_q(nn_inputs,self.eval_hidden_joint) 
            else:
                #for the case to evaluate soft_actions, not learning case
                nn_inputs = torch.cat([nn_inputs, soft_actions[:,transition_idx,:]],dim=-1)
                jq_eval, soft_actions_hidden= self.eval_joint_q(nn_inputs, soft_actions_hidden) 
            joint_q_evals.append(jq_eval)
        jq_evals = torch.stack(joint_q_evals, dim = 1)
        return jq_evals
    
    def init_hidden(self,episode_num):
        self.eval_hidden,self.target_hidden = [],[]
        for agent_id in range(self.n_agents):
            self.eval_hidden.append(self.eval_net[agent_id].init_hidden().expand(episode_num, -1))
            self.target_hidden.append(self.target_rnn[agent_id].init_hidden().expand(episode_num, -1))
        self.eval_hidden_joint = self.eval_joint_q.init_hidden().expand(episode_num, -1)

        
    def save_model(self,train_step, agent_id = None):
        num = str(train_step // self.args.save_cycle)
        for agent_id in range(self.n_agents):
            torch.save(self.eval_net[agent_id].state_dict(), os.path.join(self.output_dir, '{}_rnn-agent{}.pkl'.format(num,agent_id)))
        torch.save(self.eval_joint_q.state_dict(), os.path.join(self.output_dir, '{}_joint_q_params.pkl'.format(num)))
        
    def load_model(self,folder, num):
        for agent_id in range(self.n_agents):
            rnn_file_name = '{}/{}_rnn-agent{}.pkl'.format(folder,num,agent_id)
            self.eval_net[agent_id].load_state_dict(torch.load(rnn_file_name))
        joint_q_file = '{}/{}_joint_q_params.pkl'.format(folder,num)
        self.eval_joint_q.load_state_dict(torch.load(joint_q_file))