import numpy as np
from scipy import optimize
from types import SimpleNamespace

class RobinsonModel():
    def __init__(self):

        # Set parameters
        self.par = SimpleNamespace()
        par = self.par

        par.alpha = 0.67
        par.rho = 1.5
        par.eta = 0.1
        par.nu = 0.33

        # Set up solution containers
        self.sol = SimpleNamespace()
        sol = self.sol
        sol.w_eq = None     # Equilibrium wage
        sol.c_eq = None     # Equilibrium consumption
        sol.l_eq = None     # Equilibrium labor supply
        sol.pi_eq = None    # Equilibrium profits


    # Consumer side
    def utility(self, c, l):
        par = self.par
        return c**(1-par.rho)/(1-par.rho) - par.eta*l**(1+par.nu)/(1+par.nu)
    
    def budget_constraint(self,c, l, w, pi):
        return w*l + pi - c
    
    def consumer_optimum(self, w,pi):
        par = self.par

        l0 = 0.5
        c0 = w*0.5 

        obj = lambda x: -self.utility(x[0],x[1])            # utility function
        bc = lambda x: self.budget_constraint(x[0],x[1],w, pi)   # budget constraint
        bounds = [(0,None),(0,None)]                        # must work some and consume some

        c = optimize.minimize(obj, [c0,l0], 
                                constraints={'type':'eq', 'fun': bc},
                                bounds=bounds)
        assert c.success
        return c.x
    

    # Producer side
    def produce(self,l):
        par = self.par
        return l**par.alpha

    def profit(self,l,w):
        return self.produce(l) - w*l

    def firm_optimum(self,w):
        l0 = 0.5
        l = optimize.minimize(lambda x: -self.profit(x, w), l0, bounds=[(0,None)])
        assert l.success
        l = l.x[0]

        c = self.produce(l)
        return c, l
    

    # Equilibrium
    def excess_labor_demand(self,w):
        _, l_dem = self.firm_optimum(w) 
        pi = self.profit(l_dem, w)
        _, l_sup = self.consumer_optimum(w,pi)
        return l_dem - l_sup
    

    def find_equilibrium_wage(self, w_low, w_high, do_grid_search = True, do_print = True):
        par = self.par
        sol = self.sol

        wages = np.linspace(0.1, 2, 15)

        if do_grid_search:
            found_bracket = False
            for i,w in enumerate(wages):
                excess = self.excess_labor_demand(w)
                if do_print:
                    print(f'w = {w:.2f}, excess = {excess:.2f}')

                # save the bracket that contains 0
                if excess < 0 and not found_bracket:
                    w_low = wages[i-1]
                    w_high = wages[i]
                    found_bracket = True
        
        # Find the equilibrium wage
        w_eq = optimize.brentq(self.excess_labor_demand, w_low, w_high)
        if do_print:
            print(f'\nEquilibrium wage: {w_eq:.2f}')

        # Store solution
        sol.w_eq = w_eq
        sol.c_eq, sol.l_eq = self.firm_optimum(sol.w_eq)
        sol.pi_eq = self.profit(sol.l_eq, sol.w_eq)

