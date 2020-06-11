from misio.pacman.pacman import LocalPacmanGameRunner
import os

def load_runners():
    paths = ['pacman_layouts/' + f for f in os.listdir('pacman_layouts')]
    random_ghosts = [False, True]
    
    runners = []
    for p in paths:
        for rg in random_ghosts:
            runner = LocalPacmanGameRunner(layout_path=p,
                                           random_ghosts=rg,
                                           frame_time=0)
            runners.append(runner)
    names = [p.split('/')[1] for p in paths]
    return runners, names

