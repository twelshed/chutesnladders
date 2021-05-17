import numpy as np
from boundaries import BrownianParticle
from utils import plot_particle_paths as ppp
from utils import pos_CDFs, save_hists, CDF_gif


if __name__ == "__main__":
    delta = .25
    T = 30
    N = 100
    dt = T/N
    n_parts = 100
    alpha = 0.025
    n_steps = 10
    gif_path = 'grav/'

    bps = [BrownianParticle(2+np.random.rand(),15+np.random.rand(),N/dt, alpha = alpha) for i in range(n_parts)]
    for i in range(n_steps):
        for bp in bps:
            bp.step(dt, N/dt)

        unstuck = pos_CDFs(bps, show=False)
        save_hists(unstuck, i, gif_path)
        [bp.reset() for bp in bps]

    CDF_gif(gif_path, 'grav_gif.gif')
    #ppp(bps)
    #pos_CDFs(bps)
