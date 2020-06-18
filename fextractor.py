'''
%%%%%%%%%%%%%%%%%%%%
%    %..G.....%....%
% %%^%.%%%%%%.%.%%.%
% %.....G........%.%
% %.%%.%%  %%.%%.%.%
% .....%    %......%
% %.%%.%%%%%%.%%.%.%
% %.   ..........%.%
% %% % %%%%%%.%.%%.%
%    %    ....%...o%
%%%%%%%%%%%%%%%%%%%%
'''

import torch
from misio.pacman.game import Actions
from misio.pacman.game import Directions

'''
ghost_1
ghost_2
ghost_3
ghost_4
wall_north        bool: wall directly north
wall_east         bool: wall directly east
wall_south        bool: wall directly south
wall_west         bool: wall directly west
dist_food_north        bool: food north
dist_food_east         bool: food east
dist_food_south        bool: food south
dist_food_west         bool: food west
capsule_north
capsule_east
capsule_south
capsule_west
'''

class Extractor():
    def __call__(self, state):
        ghosts, g_states = state.getGhostPositions(), state.getGhostStates()
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules() 
        x, y = state.getPacmanPosition()
        parsed = torch.zeros((4,3), dtype=torch.float32)

        # ghosts [0th column]
        for g, s in zip(ghosts, g_states):
            dist = abs(x-g[0])+abs(y-g[1])
            if s.scaredTimer == 0 and dist <= 2:
                v = -1
            elif s.scaredTimer != 0:
                dist = cell_dist((x,y), g, walls)
                v = (s.scaredTimer-dist)/40
            else:
                v = 0

            if g[1] > y:
                parsed[0][0] = v
            if g[0] > x:
                parsed[1][0] = v
            if y > g[1]:
                parsed[2][0] = v
            if x > g[0]:
                parsed[3][0] = v

        # food distance [1st column]
        parsed[:,1] = 1     #max dist
        legals = state.getLegalActions()
        if Directions.NORTH in legals:
            parsed[0][1] = closest_food((x,y+1), food, walls) / (food.width*food.height)
        if Directions.EAST in legals:
            parsed[1][1] = closest_food((x+1,y), food, walls) / (food.width*food.height)
        if Directions.SOUTH in legals:
            parsed[2][1] = closest_food((x,y-1), food, walls) / (food.width*food.height)
        if Directions.WEST in legals:
            parsed[3][1] = closest_food((x-1,y), food, walls) / (food.width*food.height)

        # capsule distance [2nd column]
        for c in capsules:
            dist = cell_dist((x,y), c, walls) / (walls.width * walls.height)
            if c[1] > y:
                parsed[0][2] = dist
            if c[0] > x:
                parsed[1][2] = dist
            if y > c[1]:
                parsed[2][2] = dist
            if x > c[0]:
                parsed[3][2] = dist

        return parsed.cuda()

    def empty(self):
        return torch.zeros((4,3), dtype=torch.float32).cuda()


def closest_food(pos, food, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None

def cell_dist(pos, cell, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if pos_x == cell[0] and pos_y == cell[1]: 
            return dist
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    raise Exception("Cell not found!")




        
