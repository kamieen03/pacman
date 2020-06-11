import random
import torch

CAPACITY = 10000
BATCH_SIZE = 32

class ReplayMemory:
    def __init__(self):
        self.buffer = []

    def push(self, state, action, reward, next_state):
        if len(self.buffer) > CAPACITY:
            self.buffer.pop()
        self.buffer.append((state,action,reward,next_state))
    
    def sample(self):
        if len(self.buffer) < 500:
            return None
        tuples = random.choices(self.buffer, k=BATCH_SIZE)
        tuples = list(zip(*tuples))
        states = torch.stack(tuples[0]).cuda()
        actions = torch.tensor(tuples[1]).cuda()
        try:
            rewards = torch.tensor(tuples[2]).cuda()
        except:
            print(tuples[2])
        next_states = list(tuples[3])
        return states, actions, rewards, next_states

