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
    import sys
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    
    import time
    # sigma = dt/n_steps
    # x = sigma * np.sqrt(np.arange(30000)/np.pi) + np.sqrt(delta*dt)
    # xi = .33 * x
    # plt.fill_between(np.arange(30000), (x-xi), (x+xi), color= 'b', alpha=.1)

    #for t in range(7):
        #rconfig.sigma = rconfig.sigmaStep
        
        #for s in range(20):
        
    rconfig = Config();
    rconfig.sigma = rconfig.sigma * int(sys.argv[1])
    print ('rconfig.sigma:', rconfig.sigma)
    rconfig.T = int(sys.argv[2])
    rconfig.N = int(sys.argv[2])  
    print ('rconfig.T rconfig.N:', rconfig.T, rconfig.N)
       
    p = Pool(rconfig.workers)
    
    st = time.time()
    
    dnoise = rconfig.sigma*100

    bps = [BrownianParticle(rconfig,
                            rconfig.env_tuple[1]/2,
                            rconfig.env_tuple[3]/2) 
                            for i in range(rconfig.n_parts)]

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
    avgp = [bp.avg_pos for bp in bps]
  
    avgp = np.vstack(avgp)

    plt.hist2d(avgp[:,0], avgp[:,1], bins=(300, 300),range=[[0,rconfig.env_tuple[1]],[0,rconfig.env_tuple[3]]], cmap=plt.cm.jet)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Particle displacement \n Random Walks : {rconfig.n_steps*rconfig.N} \n delta = {rconfig.delta} \n Positional Noise = {dnoise}')
    plt.colorbar()
    plt.savefig(f'avgp_hist{rconfig.env_tuple[1]*100}_{rconfig.N}_{rconfig.sigma}.png')
    plt.close()



    # exp_r = [bp.exp_r for bp in bps]
    # exp_r = np.asarray(exp_r)
    # exp_r = np.sum(exp_r,axis=0)/rconfig.n_parts

    print(f'T={rconfig.T} N={rconfig.N} Sigma={rconfig.sigma} Took:{time.time()-st}')
    # plt.plot(list(range(rconfig.n_steps)),exp_r)
    # plt.xlabel(f'Random Walks : {rconfig.n_steps*rconfig.N}')
    # plt.ylabel('<r^2>')
    # plt.title(f'Expected radius N_Particles = {rconfig.n_parts} \n delta = {rconfig.delta} \n Positional Noise = {dnoise}')
    # plt.savefig(f'exp_radius{rconfig.n_steps*rconfig.N}.png')
   
            #rconfig.sigma += rconfig.sigmaStep
            
        #rconfig.T += rconfig.TNStep
        #rconfig.N += rconfig.TNStep
        
    # for bp in bps:
       # bp.step(rconfig.dt, rconfig.N)
    # ppp(bps)
    
    
    
    #pos_CDFs(bps)
