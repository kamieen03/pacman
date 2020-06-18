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


def ghost_value(dist, walls, stimer):
    return dist/(walls.width*walls.height)


class Extractor():
    def __call__(self, state):
        ghosts, g_states = state.getGhostPositions(), state.getGhostStates()
        gmap = {g: s.scaredTimer for g,s in zip(ghosts, g_states)}
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules() 
        x, y = state.getPacmanPosition()
        parsed = torch.zeros((4,4), dtype=torch.float32)
        legals = state.getLegalActions()
        
        # ghosts [0th column]
        parsed[:,0] = 1
        scared_ghosts = [g for g,s in zip(ghosts, g_states) if s.scaredTimer > 2]
        if scared_ghosts:
            if Directions.NORTH in legals:
                g, dist = closest_cell((x,y+1), scared_ghosts, walls)
                parsed[0][0] = ghost_value(dist, walls, gmap[g])
            if Directions.EAST in legals:
                g, dist = closest_cell((x+1,y), scared_ghosts, walls)
                parsed[1][0] = ghost_value(dist, walls, gmap[g])
            if Directions.SOUTH in legals:
                g, dist = closest_cell((x,y-1), scared_ghosts, walls)
                parsed[2][0] = ghost_value(dist, walls, gmap[g])
            if Directions.WEST in legals:
                g, dist = closest_cell((x-1,y), scared_ghosts, walls)
                parsed[3][0] = ghost_value(dist, walls, gmap[g])

        # food distance [1st column]
        parsed[:,1] = 1     #max dist
        if Directions.NORTH in legals:
            parsed[0][1] = closest_food((x,y+1), food, walls) / (food.width*food.height)
        if Directions.EAST in legals:
            parsed[1][1] = closest_food((x+1,y), food, walls) / (food.width*food.height)
        if Directions.SOUTH in legals:
            parsed[2][1] = closest_food((x,y-1), food, walls) / (food.width*food.height)
        if Directions.WEST in legals:
            parsed[3][1] = closest_food((x-1,y), food, walls) / (food.width*food.height)

        # capsule distance [2nd column]
        parsed[:,2] = 1     #max dist
        if capsules:
            if Directions.NORTH in legals:
                c, dist = closest_cell((x,y+1), capsules, walls)
                parsed[0][2] = dist / (food.width*food.height)
            if Directions.EAST in legals:
                c, dist = closest_cell((x+1,y), capsules, walls)
                parsed[1][2] = dist / (food.width*food.height)
            if Directions.SOUTH in legals:
                c, dist = closest_cell((x,y-1), capsules, walls)
                parsed[2][2] = dist / (food.width*food.height)
            if Directions.WEST in legals:
                c, dist = closest_cell((x-1,y), capsules, walls)
                parsed[3][2] = dist / (food.width*food.height)

        # ghosts very close [3rd column]
        for g, s in zip(ghosts, g_states):
            dist = abs(g[0]-x) + abs(g[1]-y)
            v = -1
            if s.scaredTimer > 2:
                v = 1
            if dist <= 2:
                if g[1] > y:
                    parsed[0][3] = v
                if g[0] > x:
                    parsed[1][3] = v
                if y > g[1]:
                    parsed[2][3] = v
                if x > g[0]:
                    parsed[3][3] = v
        return parsed.cuda()

    def empty(self):
        return torch.zeros((4,4), dtype=torch.float32).cuda()


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

def closest_cell(pos, cells, walls):
    dists = {c: -1 for c in cells}
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if (pos_x,pos_y) in cells:
            if dists[(pos_x,pos_y)] == -1:
                dists[(pos_x,pos_y)] = dist
            if all([v != -1 for v in dists.values()]):
                break
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    return min(list(dists.items()), key = lambda cd: cd[1])




        

