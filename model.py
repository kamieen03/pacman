import torch
from torch.nn import Linear, LeakyReLU, BatchNorm1d
import numpy as np

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = Linear(17,32)
        self.linear2 = Linear(32,64)
        self.linear3 = Linear(64,64)
        self.linear4 = Linear(64,32)
        self.linear5 = Linear(32,4)
        self.relu = LeakyReLU(inplace=False)
    
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
        state = self.relu(state)
        state = self.linear5(state)

        state = state.squeeze()
        return state
