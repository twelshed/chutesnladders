#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:43:31 2022

@author: catherinegibson
"""
from data_analysis import *
from utils import plot_particle_paths
from pylab import plot, show, grid, xlabel, ylabel, title, axis
import matplotlib.pyplot as plt

#plot_all_part_all_pos('hCQq.npy', 30, 10000000)
#plot_all('hCQq.npy', 30, 10000000, 100000000)
#plot_stuck_unstuck_dis('derp.npy', 1000, 1000, 30)
a, bps = arr_val('derp.npy')
#breakpoint()
#print(a.shape)
print(np.max(a[:, :, 1]))
print(bps)

#plot_particle_paths(bps)
#print(bps.type)
#print(bps.shape)

def plot_particle_paths(Particles):
    fig, axs = plt.subplots(5, 2) 
    for i, ax in enumerate(axs.flat):
        #breakpoint() 
        temp = Particles[i]
        x = temp[:, :2] 
        ax.plot(x[:,0],x[:,1])
        ax.plot(x[0,0],x[0,1], 'go')
        ax.plot(x[-1,0], x[-1,1], 'ro')

    # More plot decorations.
    #axis([-1, 6, -1, 31])
    axis('auto')
    title('2D Brownian Motion')
    xlabel('x', fontsize=16)
    ylabel('y', fontsize=16)
    grid(True)
    show()
    plt.savefig('derp.png',dpi=600)

def plot_particle_paths_all(Particles):
    fig = plt.figure()
    for i in range(len(Particles)):
        #breakpoint() 
        temp = Particles[i]
        x = temp[:, :2] 
        plt.plot(x[:,0],x[:,1],alpha=.3)
        plt.plot(x[0,0],x[0,1], 'go')
        plt.plot(x[-1,0], x[-1,1], 'ro')

    # More plot decorations.
    #axis([-1, 6, -1, 31])
    axis('auto')
    title('2D Brownian Motion')
    xlabel('x', fontsize=16)
    ylabel('y', fontsize=16)
    grid(True)
    show()
    plt.savefig('derp.png',dpi=600)

plot_particle_paths(bps)
plot_particle_paths_all(bps)