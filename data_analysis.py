#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt




def plot_all_part_all_pos(npy, num_part, t_steps): #npy is a string-the .npy bps file, num_part is the number of particles, time is the number of
    file = np.load(npy) # loads file as dataset
    a = file[:, :, [0,2]] #gets rid of the y column, a array is just the x pos and stuck or unstuck
    time = np.linspace(0, t_steps, t_steps)
    for i in range(0, num_part):
        plt.plot(time, a[i, :, :1])
    plt.xlabel('Time')
    plt.ylabel('X-Pos')
    plt.show()

def plot_indiv(npy, t_steps, part_num): #plot an individual particle's position over the time interval, part_num is the particle number you wish to plot
    file = np.load(npy) # loads file as dataset
    a = file[:, :, [0,2]] #gets rid of the y column, a array is just the x pos and stuck or unstuck
    time = np.linspace(0, t_steps, t_steps)
    plt.plot(time, a[part_num, :, :1])
    plt.xlabel('Time Step')
    plt.ylabel('X-pos')
    plt.show()

def plot_stuck_unstuck_dis(npy, t_lower, t_upper, num_part): #plot the stuck and unstuck particle positions for particular timeslice
    file = np.load(npy) # loads file as dataset
    a = file[:, :, [0,2]] #gets rid of the y column, a array is just the x pos and stuck or unstuck
    stuck = []
    unstuck = []
    for i in range(t_lower, t_upper-1):
        for j in range(0, num_part):
            if a[j, i, 1] == 0:
                unstuck.append(a[j, i, 0])
            elif a[j, i, 1] == 1:
                stuck.append(a[j, i, 0])
    plt.hist(unstuck, histtype= 'bar', bins= 10000)
    plt.xlabel('X-pos')
    plt.ylabel('Freq')
    plt.show()
    plt.hist(stuck, histtype= 'bar', bins= 10000)
    plt.xlabel('X-pos')
    plt.ylabel('Freq')
    plt.show()

def plot_all(npy, num_part, t_lower, t_upper): #plot all particle positions (stuck or unstuck) for a timeslice
    file = np.load(npy) # loads file as dataset
    a = file[:, :, [0,2]] #gets rid of the y column, a array is just the x pos and stuck or unstuck
    all_part = []
    for i in range(0, num_part):
        b = a[i, t_lower:t_upper, :1] #build an array for each particle for this particular time slice
        all_part.append(b)
    pos = np.vstack(all_part)
    print(len(pos))
    plt.hist(pos, histtype= 'bar', bins= 10000)
    plt.xlabel('X-pos')
    plt.ylabel('Freq')
    plt.show()

def arr_val(npy):
    file = np.load(npy) # loads file as dataset
    a = file[:, :, [0,2]]
    return a, file
