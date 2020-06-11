#!/usr/bin/env python3

import torch
import random
import sys
import numpy as np

from misio.pacman.agents import Agent
from misio.pacman.keyboardAgents import KeyboardAgent
from misio.pacman.pacman import LocalPacmanGameRunner
from misio.pacman.game import Directions

from fextractor import Extractor
from model import DQN
from memory import ReplayMemory
from utils import load_runners

EPOCHS = 500
GAMES_PER_EPOCH = 10
SAMPLES_PER_GAME = 50
EPSILON = 0.5
GAMMA = 0.98

ACTIONS = [Directions.NORTH,
           Directions.EAST,
           Directions.SOUTH,
           Directions.WEST,
           Directions.STOP]
ACTION_MAP = {d: idx for (idx, d) in enumerate(ACTIONS)}

class QAgent(Agent):
    def __init__(self):
        self.fex = Extractor()
        self.net = DQN()
        try:
            self.net.load_state_dict(torch.load('model.pth'))
        except:
            print("Starting with new weights")
        self.net.cuda()
        self.net.eval()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(),lr=1e-3)
        self.memory = ReplayMemory()
        self.training = False

        self.s = None
        self.a = None
        self.score = None

    def registerInitialState(self, state):
        self.s = None
        self.a = None
        self.score = None

    def getAction(self, game_state):
        legal = game_state.getLegalPacmanActions()
        state = self.fex(game_state)
        with torch.no_grad():
            scores = self.net(state)
        scores = list(zip(ACTIONS, scores))
        legal_scores = [p for p in scores if p[0] in legal]
        action = max(legal_scores, key = lambda p: p[1])[0]

        if self.training:
            if random.random() < EPSILON:
                action = random.choice(legal)
            if self.s is not None:
                reward = game_state.getScore() - self.score
                reward = process_reward(reward)
                self.memory.push(self.s, self.a, reward, state)
            self.s = state
            self.a = ACTION_MAP[action]
            self.score = game_state.getScore()
        return action

    def final(self, state):
        if self.training:
            reward = state.getScore() - self.score
            reward = process_reward(reward)
            self.memory.push(self.s, self.a, reward, None)


    def train(self):
        global EPSILON
        self.training = True
        runners, names = load_runners()

        for epoch in range(EPOCHS):
            print(f'Epoch {epoch} | EPSILON {EPSILON}')
            g_dict = {}
            for runner, name in zip(runners, names):
                games = []
                for game_idx in range(GAMES_PER_EPOCH):
                    game = runner.run_game(self)
                    games.append(game)
                    for _ in range(SAMPLES_PER_GAME):
                        self.training_iteration()

                avg = np.mean([game.state.getScore() for game in games])
                wins = sum([game.state.isWin() for game in games])
                print(f'{name}: {avg:0.2f} | {wins}/{GAMES_PER_EPOCH}')
            print()
            EPSILON = (np.cos(epoch*2*np.pi/20)+1)/4+0.01
            torch.save(self.net.state_dict(), 'model.pth')


    def training_iteration(self):
        # sample mini-batch
        sars = self.memory.sample()
        if sars is None:
            return
        else:
            states, actions, rewards, next_states = sars

        # replace deaths (None) with zeros and save death indeces
        deaths = []
        for i, s in enumerate(next_states):
            if s is None:
                deaths.append(i)
                next_states[i] = torch.zeros((12,)).cuda()
        next_states = torch.stack(next_states) 
        # get max Q(s',a'); deaths get value 0
        with torch.no_grad():
            next_actions_values = self.net(next_states)
            for i in deaths:
                next_actions_values[i] = torch.zeros((5,))
            best_actions_values = next_actions_values.max(dim=1).values # check
        
            # compute target values
            targets = rewards + GAMMA*best_actions_values

        # compute current action values
        actions = actions.reshape(len(actions),1)
        self.net.train()
        action_values = self.net(states).gather(1,actions).reshape(32)
        self.net.eval()
        
        # compute loss and backpropagate it
        loss = self.criterion(targets, action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def process_reward(rew):
    if rew == 9:
        return 1.0
    elif rew == 199:
        return 5
    elif rew == -1:
        return -0.1
    elif rew == -501:
        return -10
    elif rew == -491:
        return -10
    return 0

def main():
    agent = QAgent()
    if sys.argv[1] in ['t', 'train', '-t', '--train']:
        agent.train()


if __name__ == '__main__':
    main()
