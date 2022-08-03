import random
import torch
import numpy as np

class Torch_Replay_Buffer(object):
    """Replay buffer (in torch datatype) to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, obs_shape, seed, device=None):

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.observations = torch.empty([buffer_size]+[dim for dim in obs_shape],dtype=torch.float32,device=self.device)
        self.actions = torch.empty(buffer_size,dtype=torch.int64,device=self.device)
        self.rewards = torch.empty(buffer_size,dtype=torch.float32,device=self.device)
        self.dones = torch.empty(buffer_size,dtype=torch.int8,device=self.device)
        self.next_observations = torch.empty([buffer_size]+[dim for dim in obs_shape],dtype=torch.float32,device=self.device)    
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.push_count = 0
        self.obs_shape = obs_shape

    def position(self):
        return self.push_count % self.buffer_size

    def add_experience(self, obs, actions, rewards, next_obs, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            if isinstance(actions,list): exp_size = len(actions)
            else: exp_size = 1
            self.observations[self.position():self.position()+exp_size] = torch.from_numpy(obs)
            self.actions[self.position():self.position()+exp_size] = actions
            self.rewards[self.position():self.position()+exp_size] = rewards
            self.next_observations[self.position():self.position()+exp_size] = torch.from_numpy(next_obs)
            self.dones[self.position():self.position()+exp_size] = dones
        else:
            if isinstance(actions,list): exp_size = len(actions)
            else: exp_size = 1
            self.observations[self.position():self.position()+exp_size] = torch.from_numpy(obs)
            self.actions[self.position():self.position()+exp_size] = actions
            self.rewards[self.position():self.position()+exp_size] = rewards
            self.next_observations[self.position():self.position()+exp_size] = torch.from_numpy(next_obs)
            self.dones[self.position():self.position()+exp_size] = dones
        self.push_count += exp_size
   
    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        pick_range = min(self.push_count,self.buffer_size)
        random_index = random.sample(range(pick_range,k=batch_size))
        obs = self.observations[np.array(random_index)]
        actions = self.actions[np.array(random_index)]
        rewards = self.rewards[np.array(random_index)]
        next_obs = self.next_observations[np.array(random_index)]
        dones = self.dones[np.array(random_index)]
        return obs, actions, rewards, next_obs, dones
    
    def sample_seq(self, seq_len, num_experiences = None):
        """Sample sequences experience from the replay buffer"""
        def rearrange_tensor(input_tensor):
            final_tensor = torch.empty(input_tensor.size())
            current_position = self.push_count % self.buffer_size
            print(current_position)
            final_tensor[-current_position:] = input_tensor[:current_position]
            final_tensor[:self.buffer_size-current_position] = input_tensor[-self.buffer_size+current_position:]
            return final_tensor
        # if the number experience is not given
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        # compare push count and buffer size
        if self.push_count > self.buffer_size:
            ## rearrange the replay buffer
            observations = rearrange_tensor(self.observations)
            actions = rearrange_tensor(self.actions)
            rewards = rearrange_tensor(self.rewards)
            next_observations = rearrange_tensor(self.next_observations)
            dones = rearrange_tensor(self.dones)
        else: # push count is smaller than buffer size
            ## split the memory into sequence
            observations = self.observations[:self.push_count]
            actions = self.actions[:self.push_count]
            rewards = self.rewards[:self.push_count]
            next_observations = self.next_observations[:self.push_count]
            dones = self.dones[:self.push_count]
        
        valid_range = min(self.push_count,self.buffer_size)
        max_seq_num = int(valid_range/seq_len)
        valid_range = max_seq_num * seq_len
        observations = observations[:valid_range].view((max_seq_num,seq_len)+self.obs_shape)
        actions = actions[:valid_range].view((max_seq_num,seq_len,1))
        rewards = rewards[:valid_range].view((max_seq_num,seq_len,1))
        dones = dones[:valid_range].view((max_seq_num,seq_len,1))
        next_observations = next_observations[:valid_range].view((max_seq_num,seq_len)+self.obs_shape)

        # randomly sampe
        random_index = random.sample(range(max_seq_num),k=batch_size)
        obs_seqs = observations[np.array(random_index)].to(self.device) # shape: (1,seq_len,(obs shape))
        action_seqs = actions[np.array(random_index)].to(self.device)
        reward_seqs = rewards[np.array(random_index)].to(self.device)
        next_obs_seqs = next_observations[np.array(random_index)].to(self.device)
        done_seq = dones[np.array(random_index)].to(self.device)

        return obs_seqs,action_seqs,reward_seqs,next_obs_seqs,done_seq
