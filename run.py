import numpy as np
from boundaries import BrownianParticle
from utils import plot_particle_paths as ppp
from utils import pos_CDFs, save_hists, CDF_gif


if __name__ == "__main__":
    delta = .25         # average walk length
    T = 30              # time in seconds
    N = 100             # overall iterations
    dt = T/N            # delta time per iteration
    n_parts = 300         # number of particles
    alpha = 0.025       # drag coef
    n_steps = 100        # snap shot iterator
    sticky = True       # implement wall binding sites
    gif_path = 'grav/'  # animated gif output directory, must be present

    bps = [BrownianParticle(np.random.rand(),15+np.random.rand(),N, sticky = sticky, alpha = alpha, delta = delta) for i in range(n_parts)]
   
    
    
    for i in range(n_steps):
        for bp in bps:
            bp.step(dt, N)

        stuck, unstuck = pos_CDFs(bps, show=False)
        save_hists(unstuck, i, gif_path, stuck = stuck, stacked=True)
        [bp.reset() for bp in bps]

    CDF_gif(gif_path, 'grav_gif.gif')
    
    
    
    #for bp in bps:
    #    bp.step(dt, N)
    #ppp(bps)
    
    
    
    #pos_CDFs(bps)
