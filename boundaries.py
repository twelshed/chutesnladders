import numpy as np
from math import sqrt
from scipy.stats import norm

class BrownianParticle():
    def __init__(self, xpos, ypos, n_iters, alpha = 1, delta = .25, hist_roll = False, mass = 1e-5): 
        #print( xpos)
        #print( ypos)
        self.x = xpos
        self.y = ypos
        self.curr_iter = 0
        self.n_iters = int(n_iters)
        self.pos_hist = np.zeros((int(n_iters),2))
        self.rolling_history = hist_roll
        self.mass = mass
        self.g = -9.8
        self.delta = delta
        self.alpha = alpha
        #[xmin, xmax, ymin, ymax]
        #self.env_tuple = (-10000,100000,0,10000)
        self.env_tuple = (0,5,0,30)

        self.pos_hist[0,:] = np.array([xpos,ypos])

    def step(self, dt, n):
        n = int(n)
        x0 = self.pos_hist[self.curr_iter,:]

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        r = norm.rvs(size=x0.shape + (n,), scale = self.delta*sqrt(dt))
        #r = r.T
        #alpha is a drag coefficient I guess, freefall when 1
        drift = self.alpha*(self.g * dt**2)/2
       
        r[1,:] = r[1,:] + drift

        out = np.empty(r.shape)
        #np.cumsum(r, axis=-1, out=out)
        out += np.expand_dims(x0, axis=-1)
      
        #out = self.rule_check(out)
        out = self.applyValues(r,out)
        print(out)
        
        self.pos_hist[self.curr_iter:self.curr_iter+n,:] = out.T
        self.curr_iter += n
        self.x = out[0,-1]
        self.y = out[1,-1]
    
    def reset(self):
        self.pos_hist[0,:] = np.array([self.x,self.y])
        self.curr_iter = 0

    def applyValues(self,r,out):
        bounds = self.env_tuple
        print(self.pos_hist.shape)
        
        out[0,0] = self.x
        out[1,0] = self.y  
        
        for i in range(1, self.pos_hist.shape[0]):
            #stick on edge and "bounce"
            if out[0,i-1]==bounds[0]:
                if r[0,i]<0:
                    r[0,i]*=-1
            if out[0,i]==bounds[1]:
                if r[0,i]>0:
                    r[0,i]*=-1  
            if out[1,i]==bounds[2]:
                if r[0,i]<0:
                    r[0,i]*=-1  
            if out[1,i]==bounds[3]:    
                if r[0,i]>0:
                    r[0,i]*=-1      
          
            # # more aggresive attempt, don't allow movement that will hit edge
            # if (r[0,i] + out[0,i-1])<bounds[0]:
                # if r[0,i]<0:
                    # r[0,i]*=-1
            # if (r[0,i] + out[0,i-1])>bounds[1]:
                # if r[0,i]>0:
                    # r[0,i]*=-1  
            # if (r[1,i] + out[1,i-1])<bounds[2]:
                # if r[0,i]<0:
                    # r[0,i]*=-1  
            # if (r[1,i] + out[1,i-1])>bounds[3]:    
                # if r[0,i]>0:
                    # r[0,i]*=-1      
          
            out[0,i] = r[0,i] + out[0,i-1]
            out[1,i] = r[1,i] + out[1,i-1] 
            
            if out[0,i]<bounds[0]:
                out[0,i] = bounds[0]
            if out[0,i]>bounds[1]:
                out[0,i] = bounds[1]
            if out[1,i]<bounds[2]:
                out[1,i] = bounds[2]
            if out[1,i]>bounds[3]:
                out[1,i] = bounds[3]
                
        return out
                
    def rule_check(self, out):
        # truncates any movement out of bounds, currently does it across the whole history
        # should only do this when particle moves out of bounds then recalc, but for now 
        # less stupid as n approaches 1, valid for n=1
        bounds = self.env_tuple
        out[0,out[0,:]<bounds[0]] = bounds[0]
        out[0,out[0,:]>bounds[1]] = bounds[1]
        out[1,out[1,:]<bounds[2]] = bounds[2]
        out[1,out[1,:]>bounds[3]] = bounds[3]
        return out
        
