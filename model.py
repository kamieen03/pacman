import torch
from torch.nn import Linear, LeakyReLU, BatchNorm1d
import numpy as np

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = Linear(14,64)
        self.linear2 = Linear(64,128)
        self.linear3 = Linear(128,64)
        self.linear4 = Linear(64,5)
        self.relu = LeakyReLU(inplace=True)
    
    def forward(self, state):
        if len(state.shape) == 1:
            state=state.unsqueeze(0)

        state = self.linear1(state)
        state = self.relu(state)
        state = self.linear2(state)
        state = self.relu(state)
        state = self.linear3(state)
        state = self.relu(state)
        state = self.linear4(state)

        state = state.squeeze()
        return state
