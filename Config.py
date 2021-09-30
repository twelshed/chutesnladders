class Config:
    delta = .05         # average walk length
    T = 30              # time in seconds
    N = 100             # overall iterations
    dt = T/N            # delta time per iteration
    n_parts = 300         # number of particles
    alpha = 0.025       # drag coef
    n_steps = 3000        # snap shot iterator
    sticky = True       # implement wall binding sites
    gif_path = 'grav/'  # animated gif output directory, must be present
    sticking_time = 3
    env_tuple = (0,1,0,30)    #[xmin, xmax, ymin, ymax]
    hist_roll = False
    mass = 1e-5
    g = 0