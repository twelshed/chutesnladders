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
import argparse

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
    parser = argparse.ArgumentParser(description='runs experiment from paramset')
    parser.add_argument('--paramset', dest='paramset_path', type=str,
                        help='paramset to use.')
    
    args = parser.parse_args()

    with open(args.paramset_path) as json_file:
        params = json.load(json_file)

    p = Pool(Config.workers)
    letters = string.ascii_letters
    batch_id = params['batch_id']
    exp_id = params['exp_id']
    lconfig = Config()
    lconfig.env_tuple = [0,1e-6,0,1e-6]
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


    jsonstr = json.dumps(lconfig.__dict__)
    json_path = 'experiments/' + batch_id + '/' + exp_id + '/' + 'params.txt'
    with open(json_path,'wt') as f:
        f.write(jsonstr)
