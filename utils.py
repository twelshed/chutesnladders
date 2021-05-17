
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, show, grid, xlabel, ylabel, title, axis
import imageio
import glob

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

def pos_CDFs(Particles, show = True):
    #fig, ax = plt.subplots(nrows = 1, ncols = 2)
    summary_part = np.vstack([path.pos_hist for path in Particles])
    
    #ax[0].hist(summary_part[:,0], bins='auto', density=True)
    unstuck = summary_part[summary_part[:,1]>0,1]
    #unstuck = summary_part[:,0]
    if show:
        plt.hist(unstuck, bins='auto', density=True)
        plt.ylim(0,1)
        plt.xlim(0,30)
        plt.show()
    
    return unstuck

def save_hists(unstuck, i, path):
    plt.hist(unstuck, bins=100, density=True)
    plt.ylim(0,1)
    plt.xlim(0,30)
    plt.title("Particle Distribution Over Time")
    plt.ylabel("Normalized Population CDF")
    plt.xlabel("Vertical Displacement (Arb.)")
    plt.savefig(f'{path}/{i}.png')
    plt.close()

def CDF_gif(path, out_name):

    with imageio.get_writer(out_name, mode='I', fps=3) as writer:
        for filename in glob.glob(path+'*.png'):

            image = imageio.imread(filename)
            writer.append_data(image)


    
