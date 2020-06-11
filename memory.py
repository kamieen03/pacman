import random
import torch
import collections

CAPACITY = 20000
BATCH_SIZE = 32

class ReplayMemory:
    def __init__(self):
        self.buffer = collections.deque(maxlen=CAPACITY)

    def push(self, state, action, reward, next_state, next_legal):
        if len(self.buffer) > CAPACITY:
            self.buffer.pop()
        self.buffer.appendleft((state,action,reward,next_state,next_legal))
    
    def sample(self):
        if len(self.buffer) < 500:
            return None
        tuples = random.choices(self.buffer, k=BATCH_SIZE)
        tuples = list(zip(*tuples))
        states = torch.stack(tuples[0]).cuda()
        actions = torch.tensor(tuples[1]).cuda()
        rewards = torch.tensor(tuples[2]).cuda()
        next_states = list(tuples[3])
        next_legals = [list(t) for t in tuples[4]]
        return states, actions, rewards, next_states, next_legals

