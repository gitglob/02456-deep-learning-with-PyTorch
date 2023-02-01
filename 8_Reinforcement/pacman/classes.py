from collections import deque
import random
from torch import sum, pow
import torch.nn as nn
import torch.optim as optim

class ReplayMemory(object):
    """Experience Replay Memory"""
    
    def __init__(self, capacity):
        #self.size = size
        self.memory = deque(maxlen=capacity)
    
    def add(self, *args):
        """Add experience to memory."""
        self.memory.append([*args])
    
    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.sample(self.memory, batch_size)
    
    def count(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-network with target network"""
    
    def __init__(self, n_outputs, learning_rate):
        super(DQN, self).__init__()
        # network
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 8,
                kernel_size = 3,
                stride = 1,
                padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60*60*8, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs)
        )
        
        # training
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        # (batch_size, 60, 60, 3)
        x = x.permute(0, 3, 1, 2)
        # (batch_size, 3, 60, 60)
        x = self.conv_layer(x) # Apply the Convolutional layer to the input
        # (batch_size, 8, 60, 60)
        x = self.fc_layer(x) # Apply the Linear layer to the reshaped tensor    
        # (batch_size, 9)
    
        return x
        
    def loss(self, q_outputs, q_targets):
        return sum(pow(q_targets - q_outputs, 2))
    
    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)