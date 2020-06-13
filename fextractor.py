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
ghost_5
ghost_6
ghost_7
ghost_8
ghost_9
ghost_10
ghost_11
ghost_12
wall_north        bool: wall directly north
wall_east         bool: wall directly east
wall_south        bool: wall directly south
wall_west         bool: wall directly west
food_north        bool: food north
food_east         bool: food east
food_south        bool: food south
food_west         bool: food west
capsule_x_vec
capsule_y_vec
'''
ghost_positions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(0,2),(0,-2),(2,0),(-2,0)]

class Extractor():

    def __call__(self, state):
        ghosts, g_states = state.getGhostPositions(), state.getGhostStates()
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules() 
        x, y = state.getPacmanPosition()
        parsed = torch.zeros((22,), dtype=torch.float32)

        # ghosts [0-11]
        for g, s in zip(ghosts, g_states):
            if s.scaredTimer != 0:
                v = 1
            else:
                v = -1
            for idx, gp in enumerate(ghost_positions):
                if (g[0]-x,g[1]-y) == gp:
                    parsed[idx] = v

        # walls [12-15]
        for idx, (dx, dy) in enumerate([(0,1), (1,0), (0,-1), (-1,0)]):
            if walls[x+dx][y+dy]:
                parsed[idx+12] = 1

        # food position [16-19]
        for idx in range(16,20):
            parsed[idx] = 1     #max dist
        legals = state.getLegalActions()
        if Directions.NORTH in legals:
            parsed[16] = closest_food((x,y+1), food, walls) / (food.width*food.height)
        if Directions.EAST in legals:
            parsed[17] = closest_food((x+1,y), food, walls) / (food.width*food.height)
        if Directions.SOUTH in legals:
            parsed[18] = closest_food((x,y-1), food, walls) / (food.width*food.height)
        if Directions.WEST in legals:
            parsed[19] = closest_food((x-1,y), food, walls) / (food.width*food.height)

        # closest capsule position [20-21]
        cap, min_dist = (x + food.width*food.height, y + food.width*food.height), 1e6
        for c in capsules:
            dist = abs(c[0]-x) + abs(c[1]-y)
            if dist < min_dist:
                min_dist = dist
                cap = c
        parsed[20] = (cap[0] - x) / (food.width*food.height)
        parsed[21] = (cap[1] - y) / (food.width*food.height)

        return parsed.cuda()


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




        
