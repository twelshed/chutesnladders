import numpy as np
from boundaries import BrownianParticle
from utils import plot_particle_paths as ppp
from utils import pos_CDFs, save_hists, CDF_gif, expectation_radius
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from Config import Config
from sklearn.model_selection import ParameterGrid
from sklearn.utils.fixes import loguniform
import string
import random
import json

def run_index(bp):
    bp.run()
    return bp

def score(iConfig, bps):
    stucks = np.vstack([bp.stuck_hist for bp in bps])
    pos = np.vstack([bp.pos_hist for bp in bps])

    right = np.argwhere(pos[:,0]>iConfig.env_tuple[1]/2)
    left = np.argwhere(pos[:,0]<iConfig.env_tuple[1]/2)
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
    letters = string.ascii_letters
    batch_id = ''.join(random.choice(letters) for i in range(4))

    for i, params in enumerate(griditer):
    
        #### TODO write out params as json and either run aws process or loop through files in powershell script
    
        #### TODO non-looped, needs to read from command line per zip file I sent over, apparently that parameter library is really important for that
        exp_id = ''.join(random.choice(letters) for i in range(6))
        lconfig = Config()
        lconfig.env_tuple = [0,6,0,6]
        lconfig.env_tuple[1] = 1e-6
        lconfig.env_tuple[3] = params['env_y']
        lconfig.sticking_time = params['sticking_time']
        lconfig.stick_mag = params['stick_mag']
        lconfig.membrane = params['membrane']
        lconfig.exp_id = exp_id
        lconfig.batch_id = batch_id


        dnoise = lconfig.sigma*100

        bps = [BrownianParticle(lconfig,
                                np.random.uniform(0,lconfig.env_tuple[1]),
                                np.random.uniform(0,lconfig.env_tuple[3]),
                                ) 
                                for i in range(lconfig.n_parts)]

        bps = p.map(run_index, bps)

        #### TODO I think this needs to be read in right?  As part of the final operation?
        fitness[i] = score(lconfig,bps)
        X[i][0] = 1e-6
        X[i][1] = params['env_y']
        X[i][2] = params['sticking_time']
        X[i][3] = params['stick_mag']
        X[i][4] = 0 if params['membrane'] =='sigmoid' else 1
        jsonstr = json.dumps(lconfig.__dict__)

        #### TODO I think these writes should be fine in s3 as it should create the directory, but we'll need to run a 1 experiment test to be sure

        json_path = 'experiments/' + batch_id + '/' + exp_id + '/' + 'params.txt'
        with open(json_path,'wt') as f:
            f.write(jsonstr)
        
        print(fitness[i])
        if i>=n_samp:
            break;
            
        #### TODO We also probably want to make sure we can run in "offline" mode for testing

 
    y = np.asarray(fitness[:n_samp])
    X = X[:n_samp]
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression().fit(X, y)
    print(clf.coef_)
