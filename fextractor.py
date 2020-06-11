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

'''
ghost_north       bool: ghost 1 or 2 fields north
ghost_east        bool: ghost 1 or 2 fields east
ghost_south       bool: ghost 1 or 2 fields south
ghost_west        bool: ghost 1 or 2 fields west
dist_food_north   int: 0 if no food north else distance
dist_food_east    int: 0 if no food east else distance
dist_food_south   int: 0 if no food south else distance
dist_food_west    int: 0 if no food west else distance
wall_north        bool: wall directly north
wall_east         bool: wall directly east
wall_south        bool: wall directly south
wall_west         bool: wall directly west
'''
class Extractor():
    def __call__(self, state):
        ghosts = state.getGhostPositions()
        food = state.getFood()
        walls = state.getWalls()
        capsules = 
        x, y = state.getPacmanPosition()
        parsed = torch.zeros((12,), dtype=torch.float32)

        for g in ghosts:
            if g[0] == x and g[1] - y in [1,2]:
                parsed[0] = 1
            elif g[1] == y and g[0] - x in [1,2]:
                parsed[1] = 1
            elif g[0] == x and y - g[1] in [1,2]:
                parsed[2] = 1
            elif g[1] == y and x - g[0] in [1,2]:
                parsed[3] = 1
        for idx, (dx, dy) in enumerate([(0,1), (1,0), (0,-1), (-1,0)]):
            xf, yf = x+dx, y+dy
            while 0<=xf and xf<food.width and 0<=yf and yf<food.height:
                if food[xf][yf]:
                    parsed[idx+4] = abs(xf-x) + abs(yf-y)
                    divider = [food.height, food.width][idx%2]
                    parsed[idx+4] /= divider
                    break
                xf, yf = xf+dx, yf+dy
        for idx, (dx, dy) in enumerate([(0,1), (1,0), (0,-1), (-1,0)]):
            if walls[x+dx][y+dy]:
                parsed[idx+8] = 1
        return parsed.cuda()



        
