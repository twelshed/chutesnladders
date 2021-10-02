
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
    summary_part_stuck = np.vstack([path.pos_hist[np.argwhere(path.stuck_hist == 1)[:,0]] for path in Particles])
    summary_part_unstuck = np.vstack([path.pos_hist[np.argwhere(path.stuck_hist == 0)[:,0]] for path in Particles])
    
    #ax[0].hist(summary_part[:,0], bins='auto', density=True)

    unstuck = summary_part_unstuck[:,1]
    stuck = summary_part_stuck[:,1]

    #unstuck = summary_part[:,0]
    if show:
        plt.hist([unstuck,stuck], stacked=True, bins='auto', density=True)
        plt.ylim(0,1)
        plt.xlim(0,30)
        plt.show()
    
    return stuck, unstuck

def save_hists(unstuck, i, path, stuck = None, stacked = True):
    if stacked:
        plt.hist([unstuck,stuck], stacked=True, bins=100, color=["red", "blue"], label=["unstuck","stuck"])
    else:
        plt.hist(unstuck, bins=100, density=True)
    plt.ylim(0,1000)
    plt.xlim(-5,35)
    plt.legend()
    plt.title(f"Particle Distribution Over Time step:{i}")
    plt.ylabel("Normalized Population CDF")
    plt.xlabel("Vertical Displacement (Arb.)")
    plt.savefig(f'{path}/{i:04d}.png')
    plt.close()

def CDF_gif(path, out_name):

    with imageio.get_writer(out_name, mode='I', fps=30) as writer:
        for filename in sorted(glob.glob(path+'*.png')):
            print(filename)
            image = imageio.imread(filename)
            writer.append_data(image)

def expectation_radius(pos_hist, center, avg_iters=10):
    '''
        Calculates the moving average of total particle displacement from origin
        Args:
            Particles : List
            n_parts : INT number of particles
            avg_iters : INT number of iterations to take the moving average over
        Returns:
            Array of expectation values and iteration numbers

    ''' 
    exp_r = 0

    x_disp = pos_hist[:,0]
    y_disp = pos_hist[:,1]

    exp_r = np.sqrt((x_disp[-1]-center[0])**2 + (y_disp[-1]-center[1])**2)


    return exp_r







    
