import numpy as np
from boundaries import BrownianParticle
from utils import plot_particle_paths as ppp
from utils import pos_CDFs, save_hists, CDF_gif, expectation_radius
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from Config import Config

def run_index(bp):
    bp.run()
    return bp

if __name__ == "__main__": 
    #bps = [BrownianParticle(Config, np.random.rand(),Config.env_tuple[3]/2+np.random.rand()) for i in range(Config.n_parts)]

    import time
    st = time.time()
    p = Pool(Config.workers)
    # sigma = dt/n_steps
    # x = sigma * np.sqrt(np.arange(30000)/np.pi) + np.sqrt(delta*dt)
    # xi = .33 * x
    # plt.fill_between(np.arange(30000), (x-xi), (x+xi), color= 'b', alpha=.1)

    for i in range(Config.ensembles):
        dnoise = Config.sigma*100

        bps = [BrownianParticle(Config,
                                5+np.random.randn()*dnoise,
                                5+np.random.randn()*dnoise) 
                                for i in range(Config.n_parts)]

        bps = p.map(run_index, bps)


        # exp_r=np.zeros(n_steps)
        # for i in range(n_steps):

        #     for bp in bps:
        #         bp.step(dt, N)

        #     #stuck, unstuck = pos_CDFs(bps, show=False)
        #     #save_hists(unstuck, i, gif_path, stuck = stuck, stacked=True)
        #     exp_r[i] = expectation_radius(bps, n_parts,center=[5,5])

        #     [bp.reset() for bp in bps]

        #CDF_gif(gif_path, 'grav_gif.gif')
        #exp_r = np.zeros(len(bps[0].exp_r))
        exp_r = [bp.exp_r for bp in bps]
        exp_r = np.asarray(exp_r)
        exp_r = np.sum(exp_r,axis=0)/Config.n_parts

        print(f'Took:{time.time()-st}')
        plt.plot(list(range(Config.n_steps)),exp_r)
        plt.xlabel(f'Random Walks : {Config.n_steps*Config.N}')
        plt.ylabel('<r^2>')
        plt.title(f'Expected radius N_Particles = {Config.n_parts} \n delta = {Config.delta} \n Positional Noise = {dnoise}')
        plt.savefig(f'exp_radius{Config.n_steps*Config.N}.png')
   
    
    
    # for bp in bps:
       # bp.step(Config.dt, Config.N)
    # ppp(bps)
    
    
    
    #pos_CDFs(bps)
