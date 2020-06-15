import torch
from torch.nn import Linear, LeakyReLU, BatchNorm1d
import numpy as np

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear = Linear(12,4,bias=False)
    
    def forward(self, state):
        if len(state.shape) == 1:
            state=state.unsqueeze(0)

        state = self.linear(state)

        state = state.squeeze()
        return state
