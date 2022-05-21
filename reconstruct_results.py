import numpy as np
import argparse
import glob
import os
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def sliding_fitness(env_tuple, bps, recon_raw, window = 1000, step = 50):
    #stucks = np.vstack([bp[:,2] for bp in bps])
    #pos = np.vstack([bp[:,:2] for bp in bps])
    bps = np.asarray(bps).astype('uint8')

    windows = np.arange(0,bps.shape[1],step)
    fitness = np.zeros(len(windows))
    count = 0
    for i in windows:
        if recon_raw:
            inst_pos = bps[:,i:i+window,:2]
            inst_stucks = bps[:,i:i+window,2]
            right = np.argwhere(inst_pos[:,:,0]>env_tuple[1]/2)
            left = np.argwhere(inst_pos[:,:,0]<env_tuple[1]/2)
            n_right = len(right)
            n_left = len(left)

            n_stuck_right = np.sum(inst_stucks[right[:,0],right[:,1]])
            n_stuck_left = np.sum(inst_stucks[left[:,0],left[:,1]])
        else:

            inst_pos = bps[:,i:i+window,0]
            inst_stucks = bps[:,i:i+window,1]
            right = np.argwhere(inst_pos==1)
            left = np.argwhere(inst_pos==0)
            n_right = len(right)
            n_left = len(left)

            n_stuck_right = np.sum(inst_stucks[right[:,0],right[:,1]])
            n_stuck_left = np.sum(inst_stucks[left[:,0],left[:,1]])


        fitness[count] = (((n_right-n_stuck_right) - (n_left-n_stuck_left))/(n_right+n_left))
        count = count+1
    return fitness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstructs experiments')
    parser.add_argument('--exp_id', dest='exp_id', type=str,
                        help='Experiment to be reconstructed.')
    parser.add_argument('--save_figs', dest='save_figs',default = False, type=bool)
    parser.add_argument('--recon_raw', dest ='recon_raw', default = False, type=bool)
    parser.add_argument('--do_linreg', dest ='do_linreg', default = False, type=bool)
    
    args = parser.parse_args()

    exps = next(os.walk(f'experiments/{args.exp_id}'))[1]

    if not os.path.exists(f'experiments/{args.exp_id}/results'):
        os.mkdir(f'experiments/{args.exp_id}/results')
    
    env_tuple = [0,.1,0,.1]
    y = np.zeros(len(exps))
    X = np.zeros((len(exps),6))
    for i, exp in enumerate(exps):
        if exp =='results':
            continue;

        exp_dir = f'experiments/{args.exp_id}/{exp}'
        with open(f'{exp_dir}/params.txt') as f:
            d = f.read()
        if args.recon_raw:
            with open(f'{exp_dir}/params.txt') as f:
                d = f.read()

            params = json.loads(d)
            bps = [
                   np.asarray(np.loadtxt(part_file)) 
                   for part_file in 
                   set(glob.glob(f'{exp_dir}/*.txt')) - set(glob.glob(f'{exp_dir}/params.txt'))
                   ]

        else:
            params = json.loads(d)
            bps = [
                   np.load(part_file)['arr_0'] 
                   for part_file in 
                   set(glob.glob(f'{exp_dir}/*.npz')) - set(glob.glob(f'{exp_dir}/params.txt'))
                   ]

        fitness = sliding_fitness(params['env_tuple'], bps,args.recon_raw, window = 100, step = 50)
        print(fitness)
        if args.save_figs:
            plt.plot(fitness)
            plt.savefig(f'experiments/{args.exp_id}/results/{exp}.png')
            plt.close()

        #build training set
        if args.do_linreg:
            y[i] = np.average(fitness[-10:])
            X[i,0] = params['env_tuple'][1]
            X[i,1] = params['env_tuple'][3]
            X[i,2] = params['sticking_time']
            X[i,3] = params['stick_mag']
            X[i,4] = params['membrane']=='sigmoid'
            X[i,5] = params['membrane']!='sigmoid'


    #build linear model
    if args.do_linreg:
        model = LinearRegression().fit(X,y)
        print(f'score:{model.score(X,y)}')
        print(f'coefficients:{model.coef_}, intercept{model.intercept_}')


        

