import numpy as np
from scipy import optimize

class Consumer():
    def __init__(self):
        # set parameters
        self.sigma = 2.0
        self.nu = 0.01
        self.w = 1.5

    def utility(self, c, l):
        # unpack parameters
        sigma = self.sigma
        nu = self.nu

        u = (c**(1-sigma))/(1-sigma) - (l**(1+nu))/(1+nu)
        return u

    def budget_constr(self, c, l):
        w = self.w

        return w*l-c
    
    def optimize_consumer(self):
        w = self.w

        obj = lambda x: -self.utility(x[0], x[1])
        constr = lambda x: self.budget_constr(x[0], x[1])
        c0 = 4.0
        l0 = c0/w
        res = optimize.minimize(fun=obj, x0=[c0,l0], constraints={'type':'ineq', 'fun': constr}, method='SLSQP')
        return res.x[0], res.x[1]