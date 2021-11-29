class Config:
    delta = 1         # average walk length
    T = 100              # time in seconds
    N = 100             # overall iterations
    TNStep = 50
    dt = T/N            # delta time per iteration
    sigma = 1e-5 #2e-9
    sigmaStep = 1e-5 #2e-9
    n_parts = 1000         # number of particles
    alpha = 0.00       # drag coef
    n_steps = 10000        # snap shot iterator
    sticky = False       # implement wall binding sites
    gif_path = 'grav/'  # animated gif output directory, must be present
    workers = 16        # number of threads to devote to particle sim should give linear speedup with num workers
    ensembles = 1  
    sticking_time = 3
    env_tuple = (0,.1,0,.1)    #[xmin, xmax, ymin, ymax]
    hist_roll = False
    mass = 1e-5
    g = 0
    
