import numpy as np
from math import sqrt
from scipy.stats import norm

class BrownianParticle():
    def __init__(self, xpos, ypos, n_iters, sticky = False, alpha = 1, delta = .25, hist_roll = False, mass = 1e-5): 
        #print( xpos)
        #print( ypos)
        self.x = xpos
        self.y = ypos
        self.curr_iter = 0
        self.n_iters = int(n_iters)
        self.pos_hist = np.zeros((int(n_iters),2))
        self.rolling_history = hist_roll
        self.sticky = sticky
        self.mass = mass
        self.g = 0
        self.delta = delta
        self.alpha = alpha
        #currently arbitrary
        self.sticking_time = 10
        #[xmin, xmax, ymin, ymax]
        #self.env_tuple = (-10000,100000,0,10000)
        self.env_tuple = (0,1,0,30)

        self.pos_hist[0,:] = np.array([xpos,ypos])

    def step(self, dt, n):
        self.dt = dt
        x0 = self.pos_hist[self.curr_iter,:]

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        r = norm.rvs(size=x0.shape + (n,), scale = self.delta*sqrt(dt))
        #r = r.T
        #alpha is a drag coefficient I guess, freefall when 1
        drift = self.alpha*(self.g * dt**2)/2
       
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
        r = norm.rvs(size=x0.shape, scale = self.delta*sqrt(self.dt))
        #r = r.T
        #alpha is a drag coefficient I guess, freefall when 1
        drift = self.alpha*(self.g * self.dt**2)/2
       
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

    def stick_fnc(self, y, offset=7, skew=2):
        # calculates probability of a sticking event using the sigmoid function
        # offset : determines where 50% probability of sticking will be
        # skeW   : determines the ramp
        # y      : Particle vertical displacement
        # return : True if a uniform sampling event is less than or equal to the sticking probability
        p =  1 / (1 + np.exp(-y/skew - offset))

        #stickResult = np.random.rand() <= p
        stickResult = np.random.rand() <= 0.25

        print('stick_func = ' + str(stickResult) + ' p = ' + str(p))

        return stickResult

    def applyValues(self,r,out):
        bounds = self.env_tuple
        
        out[0,0] = self.x
        out[1,0] = self.y  

        sticking_time = self.sticking_time
        
        for i in range(1, self.pos_hist.shape[0]): 
            # proper bounce
            stuck = 0
            sticking_time = min(self.sticking_time,self.pos_hist.shape[0]-i)
            if (r[0,i] + out[0,i-1])<bounds[0]:
                if self.stick_fnc(out[1,i]) and self.sticky and out[0,i-1]!=bounds[0]:

                    #if r[0,i] + out[0,i-1]<bounds[0]:
                    #    print('1 out[0,i-1]='+str(out[0,i-1])+' r[0,i]='+str(r[0,i]))

                    r[0,i]= (bounds[0] - out[0,i-1])
                    r[0,i+1:i+sticking_time]= 0
                    r[1,i+1:i+sticking_time]= 0
                    stuck = sticking_time
                    
                    #if r[0,i] + out[0,i-1]<bounds[0]:
                    #    print('2 out[0,i-1]='+str(out[0,i-1])+' r[0,i]='+str(r[0,i]))
                else:
                    r[0,i]= -(out[0,i-1] - bounds[0] + (r[0,i] + out[0,i-1] - bounds[0]))
            if (r[0,i] + out[0,i-1])>bounds[1]:
                if self.stick_fnc(out[1,i]) and self.sticky and out[0,i-1]!=bounds[1]:

                    r[0,i]= (bounds[1] - out[0,i-1])
                    r[0,i+1:i+sticking_time]= 0
                    r[1,i+1:i+sticking_time]= 0
                    stuck = sticking_time
                else:
                    r[0,i]= -(out[0,i-1] - bounds[1] + (r[0,i] + out[0,i-1] - bounds[1])) 
            if (r[1,i] + out[1,i-1])<bounds[2]:
                if self.stick_fnc(out[1,i]) and self.sticky and out[0,i-1]!=bounds[2]:

                    r[1,i]= (bounds[2] - out[1,i-1] ) 
                    r[0,i+1:i+sticking_time]= 0
                    r[1,i+1:i+sticking_time]= 0 
                    stuck = sticking_time
                else:
                    r[1,i]= -(out[1,i-1] - bounds[2] + (r[1,i] + out[1,i-1] - bounds[2])) 
            if (r[1,i] + out[1,i-1])>bounds[3]: 
                if self.stick_fnc(out[1,i]) and self.sticky and out[0,i-1]!=bounds[3]:

                    r[1,i]= (bounds[3] - out[1,i-1])
                    r[0,i+1:i+sticking_time]= 0
                    r[1,i+1:i+sticking_time]= 0
                    stuck = sticking_time
                else:  
                    r[1,i]= -(out[1,i-1] - bounds[3] + (r[1,i] + out[1,i-1] - bounds[3]))       
            
            out[0,i] = r[0,i] + out[0,i-1]
            out[1,i] = r[1,i] + out[1,i-1] 
            if stuck:
                stuck = stuck - 1
                
        return out
                
        
