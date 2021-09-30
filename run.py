import numpy as np
from boundaries import BrownianParticle
from utils import plot_particle_paths as ppp
from utils import pos_CDFs, save_hists, CDF_gif
from Config import Config

if __name__ == "__main__":
    bps = [BrownianParticle(Config, np.random.rand(),Config.env_tuple[3]/2+np.random.rand()) for i in range(Config.n_parts)]
   
   
   
    for i in range(Config.n_steps):
        for bp in bps:
            bp.step(Config.dt, Config.N)

        stuck, unstuck = pos_CDFs(bps, show=False)
        save_hists(unstuck, i, Config.gif_path, stuck = stuck, stacked=True)
        [bp.reset() for bp in bps]

    CDF_gif(Config.gif_path, 'grav_gif.gif')
    
    
    
    # for bp in bps:
       # bp.step(Config.dt, Config.N)
    # ppp(bps)
    
    
    
    #pos_CDFs(bps)
