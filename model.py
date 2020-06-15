import torch
from torch.nn import Linear, LeakyReLU, BatchNorm1d
import numpy as np

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.params = torch.nn.Parameter(torch.rand(3, requires_grad = True))
    
    def forward(self, state):
        return torch.matmul(state, self.params).squeeze()
