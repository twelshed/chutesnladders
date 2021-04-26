
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, show, grid, xlabel, ylabel, title, axis

def plot_particle_paths(Particles):
    for particle in Particles: 
        x = particle.pos_hist 
        plot(x[:,0],x[:,1])
        plot(x[0,0],x[0,1], 'go')
        plot(x[-1,0], x[-1,1], 'ro')

    # More plot decorations.
    axis([-1, 6, -1, 31])
    axis('auto')
    title('2D Brownian Motion')
    xlabel('x', fontsize=16)
    ylabel('y', fontsize=16)
    grid(True)
    show()

def pos_CDFs(Particles):
    #fig, ax = plt.subplots(nrows = 1, ncols = 2)
    summary_part = np.vstack([path.pos_hist for path in Particles])
    
    #ax[0].hist(summary_part[:,0], bins='auto', density=True)
    unstuck = summary_part[summary_part[:,1]>0,1]
    plt.hist(unstuck, bins='auto', density=True)

    plt.show()


    
