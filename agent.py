#!/usr/bin/env python3

import torch
import random
import sys
import numpy as np

from misio.pacman.agents import Agent
from misio.pacman.keyboardAgents import KeyboardAgent
from misio.pacman.pacman import LocalPacmanGameRunner
from misio.pacman.game import Directions
from misio.pacman.pacman import LocalPacmanGameRunner
from misio.optilio.pacman import StdIOPacmanRunner

from fextractor import Extractor
from model import DQN
from memory import ReplayMemory
from utils import load_runners

EPOCHS = 10
GAMES_PER_EPOCH = 10
SAMPLES_PER_GAME = 200
EPSILON = 0
GAMMA = 0.98

ACTIONS = [Directions.NORTH,
           Directions.EAST,
           Directions.SOUTH,
           Directions.WEST]
ACTION_MAP = {d: idx for (idx, d) in enumerate(ACTIONS)}

class QAgent(Agent):
    def __init__(self):
        self.fex = Extractor()
        self.net = DQN()
        try:
            self.net.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        except:
            print("Starting with new weights")
            raise Exception("Weights not found")
        self.net.eval()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
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
        if Directions.STOP in legal: legal.remove(Directions.STOP)
        state = self.fex(game_state)
        if self.training:
            state = state.cuda()
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
                reward = process_reward(self.s, state, reward)
                next_legals = game_state.getLegalActions()
                if Directions.STOP in next_legals: next_legals.remove(Directions.STOP)
                next_legals = (ACTION_MAP[d] for d in next_legals)
                self.memory.push(self.s, self.a, reward, state, next_legals)
            self.s = state
            self.a = ACTION_MAP[action]
            self.score = game_state.getScore()
        return action

    def final(self, state):
        if self.training:
            reward = state.getScore() - self.score
            reward = -10
            self.memory.push(self.s, self.a, reward, None, [])


    def train(self):
        global EPSILON
        self.training = True
        self.net.cuda()
        runners, names = load_runners()

        for epoch in range(EPOCHS):
            for t in self.net.parameters():
                print(t.data)
            if epoch <= 4:
                EPSILON = [0.8, 0.5, 0.3, 0.1, 0.01][epoch]
            print('Epoch {} | EPSILON {}'.format(epoch, EPSILON))
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
                #print(f'{name}: {avg:0.2f} | {wins}/{GAMES_PER_EPOCH}')
                print('{}: {} | {}/{}'.format(name,avg, wins, GAMES_PER_EPOCH))
            print()
            torch.save(self.net.state_dict(), 'model.pth')


    def training_iteration(self):
        # sample mini-batch
        sarsl = self.memory.sample()
        if sarsl is None:
            return
        else:
            states, actions, rewards, next_states, next_state_legals = sarsl

        # replace deaths (None) with zeros
        for i, s in enumerate(next_states):
            if s is None:
                next_states[i] = self.fex.empty()
        next_states = torch.stack(next_states) 
        # get max Q(s',a'); deaths get value 0
        with torch.no_grad():
            next_actions_values = self.net(next_states)
            best_actions_values = []
            for next_legals, action_vals in zip(next_state_legals, next_actions_values):
                legal_vals = [v for (idx,v) in enumerate(action_vals) if idx in next_legals]
                if legal_vals == []:
                    legal_vals = [0]
                best_actions_values.append(max(legal_vals))
            best_actions_values = torch.tensor(best_actions_values).cuda()
        
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

    def play(self, path):
        runner = LocalPacmanGameRunner(layout_path=path,
                                       random_ghosts=True,
                                       show_window=True,
                                       zoom_window=1.0,
                                       frame_time=0.1,
                                       timeout=-1000)
        game = runner.run_game(self)


def process_reward(state, next_state, rew):
    rew = rew
    dists = state[:,1]
    next_dists = next_state[:,1]
    if dists.min() != 0 and next_dists.min() < dists.min():
        rew += 2
    else:
        dists = state[:,2]
        next_dists = next_state[:,2]
        if dists.min() != 0 and next_dists.min() < dists.min():
            rew += 2
    return rew / 50.0


def main():
    agent = QAgent()
    if len(sys.argv) > 1 and sys.argv[1] == '-t':
            agent.train()
        #else:
        #    agent.play(sys.argv[1])
    else:
        runner = StdIOPacmanRunner()
        games_num = int(input())

        for _ in range(games_num):
            runner.run_game(agent)


if __name__ == '__main__':
    main()

