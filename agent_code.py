"""

This code is based of the official pytorch implementation of TQC which can be found at https://github.com/SamsungLabs/tqc_pytorch
The paper for TQC can be found here: https://arxiv.org/abs/2005.04269


It also incorporates ideas from D2RL. The paper for this can be found here: https://arxiv.org/abs/2010.09163
and the code that this paper creates: https://github.com/pairlab/d2rl


"""

DEVICE = 'cpu'

import numpy as np
import torch
import gym
import argparse
import os
import copy
from pathlib import Path
import random
seed = 42
torch.manual_seed(seed)

random.seed(seed)
np.random.seed(seed)



  
import numpy as np
import torch
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid
from gym import spaces
import gym
  



def quantile_huber_loss(quantiles, samples, sum_over_quantiles = False):
    #return huber loss - uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise
    delta = samples[:, np.newaxis, np.newaxis, :] - quantiles[:, :, :, np.newaxis]  
    abs_delta = torch.abs(delta)
    huber_loss = torch.where(abs_delta > 1, abs_delta - 0.5, delta ** 2 * 0.5)
    n_quantiles = quantiles.shape[2]
    cumulative_prob = (torch.arange(n_quantiles, device=quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
    cumulative_prob_shaped = cumulative_prob.view(1, 1, -1, 1)
    loss = (torch.abs(cumulative_prob_shaped - (delta < 0).float()) * huber_loss)

    # Summing over the quantile dimension 
    if sum_over_quantiles:
        loss = loss.sum(dim=-2).mean()
    else:
        loss = loss.mean()

    return loss






#MLP for critic that implements D2RL architecture 
class Mlp_for_Critic(Module):
    def __init__(self,input_size,hidden_sizes,output_size):
        super().__init__()
        input_size_ = input_size
        input_dim = 28 + hidden_sizes[0] 
        self.list_of_layers = []
        for i, next_size in enumerate(hidden_sizes):
            if i == 0:
              lay = Linear(input_size_, next_size)
            else: 
              lay = Linear(input_dim, next_size)
            self.add_module(f'layer{i}', lay)
            self.list_of_layers.append(lay)
        self.last_layer = Linear(input_dim, output_size)
    

    def forward(self, input_):
        curr = input_
        for lay in self.list_of_layers:
            curr_ = relu(lay(curr))
            curr = torch.cat([curr_, input_], dim = 1)
        output = self.last_layer(curr)
        return output


#MLP for actor that implements D2RL architecture
class Mlp_for_Actor(Module):
    def __init__(self,input_size,hidden_sizes,output_size):
        super().__init__()
        self.list_of_layers = []
        input_size_ = input_size
        num_inputs = 24 
        input_dim = hidden_sizes[0] + num_inputs
        for i, next_size in enumerate(hidden_sizes):
            if i == 0:
              lay = Linear(input_size_, next_size)
            else:
              lay = Linear(input_dim, next_size)
            self.add_module(f'layer{i}', lay)
            self.list_of_layers.append(lay)
            input_size_ = next_size
            
        self.last_layer_mean_linear = Linear(input_dim, output_size)
        self.last_layer_log_std_linear = Linear(input_dim, output_size)

    def forward(self, input_):
        curr = input_

        for layer in self.list_of_layers:
            intermediate = layer(curr)
            curr = relu(intermediate)

            curr = torch.cat([curr, input_], dim=1)

        mean_linear = self.last_layer_mean_linear(curr)
        log_std_linear = self.last_layer_log_std_linear(curr)
        return mean_linear, log_std_linear





#Basic replay buffer
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size, self.ptr, self.size = max_size, 0, 0

        self.reward = np.empty((max_size, 1))
        self.state = np.empty((max_size, state_dim))
        self.reward = np.empty((max_size, 1))
        self.action = np.empty((max_size, action_dim))
        self.not_done = np.empty((max_size, 1))
        self.next_state = np.empty((max_size, state_dim))

    def sample(self, batch_size):
        #Sample from replay buffer normally - could implement ERS or PER
        index = np.random.randint(0, self.size, size=batch_size)

        r = torch.tensor(self.reward[index], dtype = torch.float, device = DEVICE)
        s = torch.tensor(self.state[index], dtype = torch.float, device = DEVICE)
        ns = torch.tensor(self.next_state[index], dtype = torch.float, device = DEVICE)
        a = torch.tensor(self.action[index], dtype = torch.float, device = DEVICE)
        nd = torch.tensor(self.not_done[index], dtype = torch.float, device = DEVICE)
        
        return s, a, ns, r, nd
 

    def add(self, state, action, next_state, reward, done):
        #Add experience to replay buffer 
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr += 1
        self.ptr %= self.max_size

        if self.max_size > self.size + 1:
          self.size = self.size + 1
        else:
          self.size = self.max_size


#Actor
class Actor(Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mlp = Mlp_for_Actor(state_dim, [512, 512], action_dim)

    def forward(self, obs):
        mean, log_std = self.mlp(obs)
        log_std = log_std.clamp(-20, 2)
        std = torch.exp(log_std)
        log_prob = None
        if self.training == False: 
            action = torch.tanh(mean)
        elif self.training == True:
            tanh_dist = TanhNormal(mean, std)
            action, pre_tanh = tanh_dist.random_sample()
            log_prob = tanh_dist.log_probability(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)      
        else:  
            print('Something wrong with training mode')
            
        return action, log_prob

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(DEVICE)[np.newaxis, :]
        action, log_prob = self.forward(obs)
        return np.array(action[0].cpu().detach())





#Critic
class Critic(Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
        super().__init__()
        self.list_of_mlp = []
        self.n_quantiles = n_quantiles

        for i in range(n_nets):
            net = Mlp_for_Critic(state_dim + action_dim, [256, 256], n_quantiles)
            self.add_module(f'net{i}', net)
            self.list_of_mlp.append(net)

    def forward(self, state, action):
        quantiles = torch.stack(tuple(net(torch.cat((state, action), dim=1)) for net in self.list_of_mlp), dim=1)
        return quantiles



class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.stand_normal = Normal(torch.zeros_like(self.normal_mean, device=DEVICE), torch.ones_like(self.normal_std, device=DEVICE))
        
        
    def logsigmoid(tensor):

      denominator = 1 + torch.exp(-tensor)
      return torch.log(1/ denominator)

    def log_probability(self, pre_tanh):
        final = (self.normal.log_prob(pre_tanh)) - (2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh))
        return final

    def random_sample(self):
        pretanh = self.normal_mean + self.normal_std * self.stand_normal.sample()
        return torch.tanh(pretanh), pretanh

import torch




class Gradient_Step(object):
  def __init__(
    self,
    *,
    actor,
    critic,
    critic_target,
    discount,
    tau,
    top_quantiles_to_drop,
    target_entropy,
    quantiles_total
  ):
    self.actor = actor
    self.critic = critic
    self.critic_target = critic_target
    self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)
    self.quantiles_total = quantiles_total
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
    
    self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
    self.discount, self.tau, self.top_quantiles_to_drop, self.target_entropy  = discount, tau, top_quantiles_to_drop,target_entropy


  def take_gradient_step(self, replay_buffer, batch_size=256):
    # Sample replay buffer
    state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
    alpha = torch.exp(self.log_alpha) #entropy temperature coefficient

    with torch.no_grad():
      # Action by the current actor for the sampled state
      new_next_action, next_log_pi = self.actor(next_state)

      ## Compute and cut quantiles at the next state
      next_z = self.critic_target(next_state, new_next_action)  
      
      #Sort and drop top k quantiles to control overestimation.
      sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
      sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

      # td error + entropy term
      target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)
    
    # Get current Quantile estimates using action from the replay buffer
    cur_z = self.critic(state, action)
    critic_loss = quantile_huber_loss(cur_z, target)


    new_action, log_pi = self.actor(state)
    # Important: detach the variable from the graph so we don't change it with other losses
    alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

    #Optimise critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Update target networks
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    # Compute actor loss
    actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()
    
    # Optimise the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    #Optimise the entropy coefficient
    self.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.alpha_optimizer.step()
  





EPISODE_LENGTH = 2000



max_timesteps = 1e6
seed = 42
n_quantiles = 25
top_quantiles_to_drop_per_net = 2
n_nets = 5
batch_size = 64
discount = 0.98
tau = 0.005


save_model = False 
# remove TimeLimit
prefix = ''
models_dir = 'logs_3'
env = gym.make('BipedalWalkerHardcore-v3')
env.seed(seed)
env.action_space.seed(seed)



state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
#Intialise everything
replay_buffer = ReplayBuffer(state_dim, action_dim)
actor = Actor(state_dim, action_dim).to(DEVICE)

critic = Critic(state_dim, action_dim, n_quantiles, n_nets).to(DEVICE)
critic_target = copy.deepcopy(critic)

top_quantiles_to_drop = top_quantiles_to_drop_per_net * n_nets

class_to_take_gradient_step = Gradient_Step(actor=actor,critic=critic,critic_target=critic_target,top_quantiles_to_drop=top_quantiles_to_drop,discount=discount,tau=tau,target_entropy=-np.prod(env.action_space.shape).item(), quantiles_total = n_quantiles * n_nets)
actor.train()
state = env.reset()
done = False
episode_timesteps = 0
episode = 1
total_num_steps = 0
ep_reward = 0
log_f = open("agent-log.txt","w+")
max_episodes = 1500
max_timesteps = 2000
for episode in range(1, max_episodes+1):
    for t in range(max_timesteps ): 
        total_num_steps += 1
        action = actor.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_timesteps += 1
    
        replay_buffer.add(state, action, next_state, reward, done)
    
        state = next_state
        ep_reward += reward
        
        if total_num_steps >= batch_size:
            class_to_take_gradient_step.take_gradient_step(replay_buffer, batch_size)
    
        if done or t==(max_timesteps-1):
            break
    
    print(f"Total T: {t + 1} Episode Num: {episode + 1} Episode T: {episode_timesteps} Reward: {ep_reward:.3f}")
    log_f.write('episode: {}, reward: {}\n'.format(episode, ep_reward))
    log_f.flush()
    ep_reward = 0
    episode += 1
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0


