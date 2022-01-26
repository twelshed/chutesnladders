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
    return ((n_right-n_stuck_right) - (n_left-n_stuck_left))/(n_right+n_left)


if __name__ == "__main__": 
    #bps = [BrownianParticle(Config, np.random.rand(),Config.env_tuple[3]/2+np.random.rand()) for i in range(Config.n_parts)]

    import time
    st = time.time()
    p = Pool(Config.workers)
    # sigma = dt/n_steps
    # x = sigma * np.sqrt(np.arange(30000)/np.pi) + np.sqrt(delta*dt)
    # xi = .33 * x
    # plt.fill_between(np.arange(30000), (x-xi), (x+xi), color= 'b', alpha=.1)
    grid = {
                  'env_y': np.linspace(1e-6, 3e-5,10),
                  'membrane': ['sigmoid','step'],
                  'stick_mag': np.linspace(0,1,10),
                  'sticking_time': np.linspace(0,100,10)}

    griditer = ParameterGrid(grid)
    fitness = [None]*len(griditer)
    X = np.zeros((len(griditer),5))
    print("Num permutations to test:" + str(len(griditer)))
    n_samp = 100

    for i, params in enumerate(griditer):
        Config.env_tuple[1] = 1e-6
        Config.env_tuple[3] = params['env_y']
        Config.sticking_time = params['sticking_time']
        Config.stick_mag = params['stick_mag']
        Config.membrane = params['membrane']

        dnoise = Config.sigma*100

        bps = [BrownianParticle(Config,
                                np.random.uniform(0,Config.env_tuple[1]),
                                np.random.uniform(0,Config.env_tuple[3]),
                                ) 
                                for i in range(Config.n_parts)]

        bps = p.map(run_index, bps)

        fitness[i] = score(Config,bps)
        X[i][0] = 1e-6
        X[i][1] = params['env_y']
        X[i][2] = params['sticking_time']
        X[i][3] = params['stick_mag']
        X[i][4] = 0 if params['membrane'] =='sigmoid' else 1
        print(fitness[i])
        if i>=n_samp:
            break;

 
    y = np.asarray(fitness[:n_samp])
    X = X[:n_samp]
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression().fit(X, y)
    print(clf.coef_)
