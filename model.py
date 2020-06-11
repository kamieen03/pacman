import torch
from torch.nn import Linear, LeakyReLU, BatchNorm1d
import numpy as np

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linears = torch.nn.ModuleList([Linear(12,64),
                                            Linear(64,128),
                                            Linear(128,64),
                                            Linear(64,5)])
#        self.bnorms = torch.nn.ModuleList([BatchNorm1d(64),
#                                           BatchNorm1d(64),
#                                           BatchNorm1d(32)])
        self.relu = LeakyReLU(inplace=True)
    
    def forward(self, state):
        if len(state.shape) == 1:
            state=state.unsqueeze(0)
        for l in self.linears[:-1]:
            state = l(state)
#            state = bn(state)
            state = self.relu(state)
        state = self.linears[-1](state)
        state = state.squeeze()
        return state
