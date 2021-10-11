import numpy as np
from boundaries import BrownianParticle
from utils import plot_particle_paths as ppp
from utils import pos_CDFs, save_hists, CDF_gif, expectation_radius
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool

def run_index(bp):
    bp.run()
    return bp


if __name__ == "__main__":
    delta = 1         # average walk length
    T = 100              # time in seconds
    N = 100             # overall iterations
    dt = T/N            # delta time per iteration
    sigma = 2e-9
    n_parts = 100         # number of particles
    alpha = 0.00       # drag coef
    n_steps = 3000        # snap shot iterator
    sticky = False       # implement wall binding sites
    gif_path = 'grav/'  # animated gif output directory, must be present
    workers = 16        # number of threads to devote to particle sim should give linear speedup with num workers

    #bps = [BrownianParticle(np.random.rand(),15+np.random.rand(),N, sticky = sticky, alpha = alpha, delta = delta) for i in range(n_parts)]
    import time
    st = time.time()
    p = Pool(workers)
    # sigma = dt/n_steps
    # x = sigma * np.sqrt(np.arange(30000)/np.pi) + np.sqrt(delta*dt)
    # xi = .33 * x
    # plt.fill_between(np.arange(30000), (x-xi), (x+xi), color= 'b', alpha=.1)
    ensembles = 10

    for i in range(ensembles):
        dnoise = sigma*100

        bps = [BrownianParticle(5+np.random.randn()*dnoise,
                                5+np.random.randn()*dnoise, 
                                N, 
                                n_steps, 
                                dt, 
                                sticky = sticky,
                                alpha = alpha, 
                                delta = delta) 
                                for i in range(n_parts)]

        



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
        exp_r = np.sum(exp_r,axis=0)/n_parts

        print(f'Took:{time.time()-st}')
        plt.plot(list(range(n_steps)),exp_r)
        plt.xlabel(f'Random Walks : {n_steps*N}')
        plt.ylabel('<r^2>')
        plt.title(f'Expected radius N_Particles = {n_parts} \n delta = {delta} \n Positional Noise = {dnoise}')
        plt.savefig(f'exp_radius{n_steps*N}.png')




    
    
    
    #for bp in bps:
    #    bp.step(dt, N)
    #ppp(bps)
    
    
    
    #pos_CDFs(bps)
