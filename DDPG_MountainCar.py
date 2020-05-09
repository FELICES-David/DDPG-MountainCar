##Imports
from collections import deque
import random
from copy import copy, deepcopy
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import wrappers
from torch.optim import Adam


##Hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE=64
GAMMA=0.99
TAU=0.001       #Target Network HyperParameters Update rate
LRA=0.0001      #LEARNING RATE ACTOR
LRC=0.001       #LEARNING RATE CRITIC
H1=400   #neurons of 1st layers
H2=300   #neurons of 2nd layers
INIT_W = 0.003  #initial weights' limit
MAX_EPISODES=5000 #number of episodes of the training
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
ENV_NAME = 'MountainCarContinuous-v0'
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

##Replay Buffer
class ReplayBuffer(object):
    def __init__(self, buffer_size = BUFFER_SIZE, name_buffer=''):
        self.buffer_size=buffer_size
        self.num_exp=0
        self.buffer=deque()

    def add(self, s, a, r, t, s2):
        experience=(s, a, r, t, s2)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size = BATCH_SIZE):
        if self.num_exp < batch_size:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, batch_size)
        s, a, r, t, s2 = map(np.stack, zip(*batch))
        return s, a, r, t, s2

    def clear(self):
        self.buffer = deque()
        self.num_exp=0


##Normalized Environment
class NormalizedEnv(gym.ActionWrapper):
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    
    
##Networks
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=H1, hidden2=H2, init_w=INIT_W):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        out = self.tanh(x)
        return out
    
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.uniform(-1.,1.,self.nb_actions)
        else:
            action = self.forward(state)
            return action

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=H1, hidden2=H2, init_w=INIT_W):
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(nb_states,hidden1)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(hidden1,hidden2)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        self.fca1 = nn.Linear(nb_actions,hidden2)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        s = F.relu(self.fcs1(state))
        s = self.relu(s)
        x = self.fc2(T.cat([s,action],1))
        x = self.relu(x)
        out = self.fc3(x)
        return out
    

##DDPG
env = NormalizedEnv(gym.make(ENV_NAME))
    
NB_states = env.observation_space.shape[0]
NB_actions = env.action_space.shape[0]
    
class DDPG(object):
    def __init__(self, nb_states = NB_states, nb_actions = NB_actions):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.critic  = Critic(self.nb_states, self.nb_actions).to(device)
        self.actor = Actor(self.nb_states, self.nb_actions).to(device)
        self.target_critic  = Critic(self.nb_states, self.nb_actions).to(device)
        self.target_actor = Actor(self.nb_states, self.nb_actions).to(device)
        self.critic_optim  = Adam(self.critic.parameters(), lr=LRC)
        self.actor_optim  = Adam(self.actor.parameters(), lr=LRA)
        self.memory = ReplayBuffer()
        self.exploration = EXPLORATION_MAX

    def update():
        batch = self.memory.sample()
        for state, action, reward, state_next, terminal in batch:
            ####not yet complete