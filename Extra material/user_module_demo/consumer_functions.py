import numpy as np
from scipy import optimize

# define utility function
def utility(c, l, sigma, nu):
    u = (c**(1-sigma))/(1-sigma) - (l**(1+nu))/(1+nu)
    return u

# define budget constraint (negative when violated)
def budget_constr(c, l, w):
    return w*l-c

# define optimization
def optimize_consumer(sigma, nu, w):
    obj = lambda x: -utility(x[0], x[1], sigma, nu)
    constr = lambda x: budget_constr(x[0], x[1], w)
    c0 = 4.0
    l0 = c0/w
    res = optimize.minimize(fun=obj, x0=[c0,l0], constraints={'type':'ineq', 'fun': constr}, method='SLSQP')
    return res.x[0], res.x[1]