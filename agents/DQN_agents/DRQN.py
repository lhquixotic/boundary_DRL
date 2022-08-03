import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Torch_Replay_Buffer import Torch_Replay_Buffer

from agents.DQN_agents.DQN import DQN


class QRNN_net(nn.Module):
    ''' RNN network as Q net'''
    def __init__(self,input_dim,output_dim,layers_info,hyperparameters=None,override_seed=None):
        super(QRNN_net,self).__init__()
        self.state_space = input_dim
        self.aciton_space = output_dim
        self.hidden_space = layers_info[-1]
        
        self.Linear1 = nn.Linear(self.state_space,128)
        self.Linear11 = nn.Linear(128,64)
        self.lstm = nn.LSTM(64,64,batch_first=True)
        self.Linear2 = nn.Linear(64,self.aciton_space)
        
    def forward(self,x,h,c):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear11(x))
        x, (new_h,new_c) = self.lstm(x,(h,c))
        x = self.Linear2(x)
        return x,new_h,new_c

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])

class DRQN(DQN):
    ''' A DQN agent that use LSTM as a layer of q net '''
    agent_name = "DRQN"
    def __init__(self, config):
        DQN.__init__(self,config)
        self.q_network_local = QRNN_net(input_dim=self.state_size,output_dim=self.action_size
                                            ,layers_info=self.hyperparameters["linear_hidden_units"]).to(self.device)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                                lr = self.hyperparameters["learning_rate"],eps=1e-4)
        self.q_network_target = QRNN_net(input_dim=self.state_size,output_dim=self.action_size,
                                            layers_info=self.hyperparameters["linear_hidden_units"]).to(self.device)
        self.q_network_target.load_state_dict(self.q_network_local.state_dict())

        # use torch replay buffer
        # print("obs_shape is {}".format(self.environment.observation_space.shape))
        print(self.environment.observation_space)
        self.memory = Torch_Replay_Buffer(self.hyperparameters["buffer_size"],self.hyperparameters["batch_size"],(self.state_size,),self.device)
        self.seq_len = 10
        self.epsilon = self.hyperparameters['epsilon']
        self.decay = self.hyperparameters['epsilon_decay_rate_denominator']
        
        print(self.q_network_local)

    def step(self):
        """ Run an episode """
        h,c = self.q_network_local.init_hidden_state(batch_size=self.hyperparameters['batch_size'],training=False)
        h = h.to(self.device)
        c = c.to(self.device)
        while not self.done:
            self.action,h,c = self.pick_action(state=self.state,h=h.to(self.device),c=c.to(self.device))
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn()
                self.update_target_net()
            self.save_experience()
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1
        # self.epsilon = max(self.epsilon*self.decay, 0.001)

    def pick_action(self, h, c, state=None,):
        ''' Considering h and c '''
        if state is None: state = self.state
        if isinstance(state,np.int64) or isinstance(state,int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        # print("State shape:{}, h shape:{}, c shape:{}.".format(state.shape,h.shape,c.shape))
        if len(state.shape) < 2: state = state.unsqueeze(0)
        # print("state shape:{}".format(state.shape))
        self.q_network_local.eval() # put the network in evaluation mode
        with torch.no_grad():
            action_values, h, c = self.q_network_local(state,h,c)
        self.q_network_local.train() # put the network in training mode
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        return action,h,c
    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
            
        Q_expected = self.compute_expected_q_values(states, actions)
        # print(Q_targets.shape,Q_expected.shape)
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        return loss

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        gamma = self.hyperparameters["discount_rate"]
        # gamma = torch.tensor(gamma).float().to(self.device)
        Q_targets_current = rewards + (gamma * Q_targets_next * (1 - dones))
        # print("c",Q_targets_current.shape)
        return Q_targets_current
        
    def compute_q_values_for_next_states(self, next_states):
        h_targets, c_targets = self.q_network_target.init_hidden_state(batch_size=self.hyperparameters["batch_size"],training=True)
        # print("State shape:{}, h shape:{}, c shape:{}.".format(next_states.shape,h_targets.shape,c_targets.shape))
        # print("State shape:{}, h shape:{}, c shape:{}.".format(next_states.device,h_targets.to(self.device).device,c_targets.to(self.device).device))
        q_targets,_,_ = self.q_network_target(next_states.to(self.device),h_targets.to(self.device),c_targets.to(self.device))
        Q_targets_next = q_targets.max(2)[0].view(self.hyperparameters["batch_size"],self.seq_len,-1).detach()
        return Q_targets_next

    def compute_expected_q_values(self, states, actions):
        hs, cs = self.q_network_local.init_hidden_state(batch_size=self.hyperparameters["batch_size"],training=True)
        q_out, _, _ = self.q_network_local(states.to(self.device),hs.to(self.device),cs.to(self.device))
        Q_expected = q_out.gather(2,actions.long())
        return Q_expected

    def sample_experiences(self):
        """Draws a random sample sequences of experience from the memory buffer"""
        experiences = self.memory.sample_seq(self.seq_len)
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones
    
    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        # if self.memory.push_count > self.hyperparameters["batch_size"]*self.seq_len: print("enough")
        return self.memory.push_count > self.hyperparameters["batch_size"]*self.seq_len

    def update_target_net(self):
        for target_para,local_para in zip(self.q_network_target.parameters(),self.q_network_local.parameters()):
            target_para.data.copy_(self.hyperparameters["tau"]*local_para + (1-self.hyperparameters["tau"])*target_para.data)