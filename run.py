import numpy as np
from Particle import BrownianParticle
from utils import plot_particle_paths as ppp
from utils import pos_CDFs


if __name__ == "__main__":
    delta = .25
    T = 100
    N = 500
    dt = T/N
    n_parts = 50
    alpha = .025

    bps = [BrownianParticle(2+np.random.rand(),15+np.random.rand(),N/dt, alpha = alpha) for i in range(n_parts)]
    for bp in bps:
        bp.step(dt, N/dt)

    #ppp(bps)
    pos_CDFs(bps)
