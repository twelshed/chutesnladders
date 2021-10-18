class Config:
    delta = .0001         # average walk length
    T = 30              # time in seconds
    N = 100             # overall iterations
    dt = T/N            # delta time per iteration
    n_parts = 100         # number of particles
    alpha = 0.00       # drag coef
    n_steps = 100        # snap shot iterator
    sticky = False       # implement wall binding sites
    gif_path = 'grav/'  # animated gif output directory, must be present
    workers = 16        # number of threads to devote to particle sim should give linear speedup with num workers
    sticking_time = 3
    env_tuple = (0,1,0,30)    #[xmin, xmax, ymin, ymax]
    hist_roll = False
    mass = 1e-5
    g = 0
    
