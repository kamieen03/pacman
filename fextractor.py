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

'''
ghost_north       tristate: ghost 1 or 2 fields north; 1 if scared else -1
ghost_east        tristate: ghost 1 or 2 fields east
ghost_south       tristate: ghost 1 or 2 fields south
ghost_west        tristate: ghost 1 or 2 fields west
food_north        bool: food north
food_east         bool: food east
food_south        bool: food south
food_west         bool: food west
wall_north        bool: wall directly north
wall_east         bool: wall directly east
wall_south        bool: wall directly south
wall_west         bool: wall directly west
capsule_north     bool: whether there is a capsule 1 or 2 fields around 
capsule_east      bool: whether there is a capsule 1 or 2 fields around 
capsule_south     bool: whether there is a capsule 1 or 2 fields around 
capsule_west      bool: whether there is a capsule 1 or 2 fields around 
closest_food      int: distance to the closest food
'''
class Extractor():
    def __call__(self, state):
        ghosts, g_states = state.getGhostPositions(), state.getGhostStates()
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules() 
        x, y = state.getPacmanPosition()
        parsed = torch.zeros((17,), dtype=torch.float32)

        # ghosts
        for g, s in zip(ghosts, g_states):
            if s.scaredTimer != 0:
                v = 1
            else:
                v = -1
            if g[0] == x:
                if g[1] - y in [1,2]:
                    parsed[0] = v
                elif y - g[1] in [1,2]:
                    parsed[2] = v
            elif g[1] == y:
                if g[0] - x in [1,2]:
                    parsed[1] = v
                elif x - g[0] in [1,2]:
                    parsed[3] = v

        # food position
        foods = check_food(parsed,x,y,food)
        for idx in range(4):
            parsed[idx+4] = foods[idx]
        for idx, (dx, dy) in enumerate([(0,1), (1,0), (0,-1), (-1,0)]):
            if walls[x+dx][y+dy]:
                parsed[idx+8] = 1

        # capsules position
        for c in capsules:
            if c[1]-y >= 0:
                parsed[12] = 1
            if c[0]-x >= 0:
                parsed[13] = 1
            if y-c[1] >= 0:
                parsed[14] = 1
            if x-c[0] >= 0:
                parsed[15] = 1

        # food distance
        parsed[16] = closest_food((x,y), food, walls)/max(walls.height, walls.width)

        return parsed.cuda()

def check_food(parsed, x, y, food):
    result = [0,0,0,0]
    yf, found = y, False
    while yf < food.height and not found:
        for xf in range(food.width):
            if food[xf][yf]:
                result[0] = 1
                found = True
                break
        yf += 1
    xf, found = x, False
    while xf < food.width and not found:
        for yf in range(food.height):
            if food[xf][yf]:
                result[1] = 1
                found = True
                break
        xf += 1
    yf, found = y, False
    while yf > 0 and not found:
        for xf in range(food.width):
            if food[xf][yf]:
                result[2] = 1
                found = True
                break
        yf -= 1
    xf, found = x, False
    while xf >= 0 and not found:
        for yf in range(food.height):
            if food[xf][yf]:
                result[3] = 1
                found = True
                break
        xf -= 1
    return result

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




        
