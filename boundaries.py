import numpy as np
from math import sqrt
from scipy.stats import norm
from threading import Thread
import string
import random
import sys
import os
from utils import expectation_radius
from numpy import savez_compressed

class BrownianParticle():
    def __init__(self, config, xpos, ypos,bp_seed): 
        #print( xpos)
        #print( ypos)
        self.Config = config
        self.env_tuple = config.env_tuple
        self.x = xpos
        self.y = ypos
        self.curr_iter = 0
        self.n_iters = int(self.Config.N)
        self.stuck_hist = np.zeros((int(self.Config.N),1))
        self.pos_hist = np.zeros((int(self.Config.N),2))
        self.exp_r = np.zeros(self.Config.n_steps)
        self.score = np.zeros(self.Config.n_steps)
        self.avg_pos = np.zeros((self.Config.n_steps,2))
        letters = string.ascii_letters
        self.part_id = ''.join(random.choice(letters) for i in range(4))
        self.exp_id = config.exp_id
        self.batch_id = config.batch_id
        self.delete_raw = config.delete_raw
        self.pos_hist[0,:] = np.array([xpos,ypos])
        self.bp_seed = bp_seed
        self.seeded = 0

        
        #print("iiConfig test = " + str(self.Config.env_tuple[1]) + " and " + str(self.Config.env_tuple[3]))
    
    def run(self):

        #print("Config test = " + str(self.Config.env_tuple[1]) + " and " + str(self.Config.env_tuple[3]))
        #print("sticking_time = " + str(self.Config.sticking_time))
        # sys.stdout.write('[%s] running ...  process id: %s\n' 
        #                  % (self.name, os.getpid()))
        for i in range(self.Config.n_steps):
            self.step(self.Config.dt,self.n_iters)
            self.avg_pos[i] = self.pos_hist[-1,:]
            self.exp_r[i] = expectation_radius(self.pos_hist, center=[.1,.1])
            self.write_results()
            if i < self.Config.n_steps-1:
                self.reset()
        self.compress_results()

    def write_results(self):

        out_dir = 'experiments/' + self.batch_id + '/' + self.exp_id+'/'

        out_path = out_dir + self.part_id+'.txt'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


        #pos_x = np.digitize(self.pos_hist[:,0], np.arange(0,self.env_tuple[1],self.env_tuple[1]/65000)) 
        #pos_y = np.digitize(self.pos_hist[:,1], np.arange(0,self.env_tuple[3],self.env_tuple[3]/65000))
        combined = np.concatenate((self.pos_hist,self.stuck_hist),axis=-1)
        
        #combined = combined.astype('float16')
        with open(out_path,"ab") as f:
            np.savetxt(f, combined)

    def compress_results(self):
        out_dir = 'experiments/' + self.batch_id + '/' + self.exp_id+'/'

        in_path = out_dir + self.part_id+'.txt'
        out_path = out_dir + self.part_id+'.npz'
        with open(in_path,"rb") as fi:
            d = np.loadtxt(fi)

        #delete y positions, set particle positions on left to 0, right to 1, save as bool array

        d = np.delete(d,1,1)


        d[:,0] = d[:,0]>(self.env_tuple[1]/2)

        with open(out_path,'wb') as fo:
            np.savez_compressed(fo,d.astype('bool'))
       
        if self.delete_raw:
            if os.path.exists(in_path):
                os.remove(in_path)

    def step_score(self,i):

        stucks = self.stuck_hist
        right = np.argwhere(self.pos_hist[:,0]>self.Config.env_tuple[1]/2)
        left = np.argwhere(self.pos_hist[:,0]<self.Config.env_tuple[1]/2)
        n_right = len(right)
        n_left = len(left)
        n_stuck_right = np.sum(stucks[right])
        n_stuck_left = np.sum(stucks[left])
        self.score[i] =  ((n_right-n_stuck_right) - (n_left-n_stuck_left))/(n_right+n_left)

    def step(self, dt, n):
        self.dt = dt
        x0 = self.pos_hist[self.curr_iter,:]

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.


        r = norm.rvs(size=x0.shape + (n,), scale = self.Config.sigma)
        #r = r.T
        #alpha is a drag coefficient I guess, freefall when 1
        drift = self.Config.alpha*(self.Config.g * dt**2)/2
       
        r[1,:] = r[1,:] + drift

        out = np.empty(r.shape)
        out += np.expand_dims(x0, axis=-1)
        out = self.applyValues(r,out)
        
        self.pos_hist[self.curr_iter:self.curr_iter+n,:] = out.T
        self.curr_iter += n
        self.x = out[0,-1]
        self.y = out[1,-1]
    
    def step_interrupt(self, curr_iter, out):
        # if there is  a sticking event we need to leave the particle there for some time then recalculate the walk
        x0 = self.pos_hist
        x0[curr_iter:curr_iter + self.sticking_time,:] = x0[curr_iter-1,:]
        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        r = norm.rvs(size=x0.shape, scale = self.Config.sigma)
        #r = r.T
        #alpha is a drag coefficient I guess, freefall when 1
        drift = self.Config.alpha*(self.Config.g * self.dt**2)/2
       
        r += drift

        out = np.empty(r.shape)
        out[:curr_iter,:] = x0[:curr_iter,:]
        out[curr_iter + self.sticking_time:,:] += r[curr_iter+self.sticking_time:,:]
        
        
        self.pos_hist = out
        return out.T

    def reset(self):
        self.pos_hist = np.zeros_like(self.pos_hist)
        self.pos_hist[0,:] = np.array([self.x,self.y])
        self.curr_iter = 0

    def stick_fnc(self, y, offset=15, skew=2):
        # calculates probability of a sticking event using the sigmoid function
        # offset : determines where 50% probability of sticking will be
        # skeW   : determines the ramp
        # y      : Particle vertical displacement
        # return : True if a uniform sampling event is less than or equal to the sticking probability
        if self.Config.membrane == 'sigmoid':
            p =  (1 / (1 + np.exp(-y/skew + offset)))*self.Config.stick_mag

        else:
            if y > self.Config.env_tuple[1]/2:
                p = self.Config.stick_mag
            else:
                p = 0
        #stickResult = np.random.rand() <= 0.25

        #print('stick_func = ' + str(stickResult) + ' p = ' + str(p))

        return np.random.rand() <= p

    def applyValues(self,r,out):
        bounds = self.Config.env_tuple
        
        out[0,0] = self.x
        out[1,0] = self.y  

        sticking_time = self.Config.sticking_time
        
        for i in range(1, self.pos_hist.shape[0]):         
            # proper bounce
            stuck = 0
            sticking_time = min(sticking_time,self.pos_hist.shape[0]-i)
            if (r[0,i] + out[0,i-1])<bounds[0]:
                if self.stick_fnc(out[1,i]) and self.Config.sticky and out[0,i-1]!=bounds[0]:

                    #if r[0,i] + out[0,i-1]<bounds[0]:
                    #    print('1 out[0,i-1]='+str(out[0,i-1])+' r[0,i]='+str(r[0,i]))

                    r[0,i]= (bounds[0] - out[0,i-1])
                    r[0,i+1:i+sticking_time]= 0
                    r[1,i+1:i+sticking_time]= 0
                    self.stuck_hist[i+1:i+sticking_time]= 1
                    stuck = sticking_time
                    
                    #if r[0,i] + out[0,i-1]<bounds[0]:
                    #    print('2 out[0,i-1]='+str(out[0,i-1])+' r[0,i]='+str(r[0,i]))
                else:
                    r[0,i]= -(out[0,i-1] - bounds[0] + (r[0,i] + out[0,i-1] - bounds[0]))
            if (r[0,i] + out[0,i-1])>bounds[1]:
                if self.stick_fnc(out[1,i]) and self.Config.sticky and out[0,i-1]!=bounds[1]:

                    r[0,i]= (bounds[1] - out[0,i-1])
                    r[0,i+1:i+sticking_time]= 0
                    r[1,i+1:i+sticking_time]= 0
                    self.stuck_hist[i+1:i+sticking_time]= 1
                    stuck = sticking_time
                else:
                    r[0,i]= -(out[0,i-1] - bounds[1] + (r[0,i] + out[0,i-1] - bounds[1])) 
            if (r[1,i] + out[1,i-1])<bounds[2]:
                if self.stick_fnc(out[1,i]) and self.Config.sticky and out[0,i-1]!=bounds[2]:

                    r[1,i]= (bounds[2] - out[1,i-1] ) 
                    r[0,i+1:i+sticking_time]= 0
                    r[1,i+1:i+sticking_time]= 0
                    self.stuck_hist[i+1:i+sticking_time]= 1                    
                    stuck = sticking_time
                else:
                    r[1,i]= -(out[1,i-1] - bounds[2] + (r[1,i] + out[1,i-1] - bounds[2])) 
            if (r[1,i] + out[1,i-1])>bounds[3]: 
                if self.stick_fnc(out[1,i]) and self.Config.sticky and out[0,i-1]!=bounds[3]:

                    r[1,i]= (bounds[3] - out[1,i-1])
                    r[0,i+1:i+sticking_time]= 0
                    r[1,i+1:i+sticking_time]= 0
                    self.stuck_hist[i+1:i+sticking_time]= 1
                    stuck = sticking_time
                else:  
                    r[1,i]= -(out[1,i-1] - bounds[3] + (r[1,i] + out[1,i-1] - bounds[3]))       
            
            out[0,i] = r[0,i] + out[0,i-1]
            out[1,i] = r[1,i] + out[1,i-1] 
            if stuck:
                stuck = stuck - 1
                
        return out
                
        
