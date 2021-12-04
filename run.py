import numpy as np
from boundaries import BrownianParticle
from utils import plot_particle_paths as ppp
from utils import pos_CDFs, save_hists, CDF_gif, expectation_radius
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from Config import Config
from sklearn.model_selection import ParameterGrid
from sklearn.utils.fixes import loguniform

def run_index(bp):
    bp.run()
    return bp

def score(Config, bps):
    stucks = np.vstack([bp.stuck_hist for bp in bps])
    pos = np.vstack([bp.pos_hist for bp in bps])

    right = np.argwhere(pos[:,0]>Config.env_tuple[1]/2)
    left = np.argwhere(pos[:,0]<Config.env_tuple[1]/2)
    n_right = len(right)
    n_left = len(left)
    n_stuck_right = np.sum(stucks[right])
    n_stuck_left = np.sum(stucks[left])
    return (n_left-n_stuck_left) / (n_right/n_stuck_right)


if __name__ == "__main__": 
    #bps = [BrownianParticle(Config, np.random.rand(),Config.env_tuple[3]/2+np.random.rand()) for i in range(Config.n_parts)]

    import time
    st = time.time()
    p = Pool(Config.workers)
    # sigma = dt/n_steps
    # x = sigma * np.sqrt(np.arange(30000)/np.pi) + np.sqrt(delta*dt)
    # xi = .33 * x
    # plt.fill_between(np.arange(30000), (x-xi), (x+xi), color= 'b', alpha=.1)
    grid = {'env_x': np.linspace(1e-5, 1e-3,10), 
                  'env_y': np.linspace(1e-3, 3e-3,10),
                  'membrane': ['sigmoid','step'],
                  'stick_mag': np.linspace(.1,.9,10),
                  'sticking_time': np.linspace(1,100,10)}

    griditer = ParameterGrid(grid)
    fitness = [None]*len(griditer)

    for i, params in enumerate(griditer):
        Config.env_tuple[1] = params['env_x']
        Config.env_tuple[3] = params['env_y']
        Config.sticking_time = params['sticking_time']
        Config.stick_mag = params['stick_mag']
        Config.membrane = params['membrane']

        dnoise = Config.sigma*100

        bps = [BrownianParticle(Config,
                                Config.env_tuple[1]/2+np.random.randn()*dnoise,
                                Config.env_tuple[3]/2+np.random.randn()*dnoise) 
                                for i in range(Config.n_parts)]

        bps = p.map(run_index, bps)

        fitness[i] = score(Config,bps)
        print(fitness[i])
