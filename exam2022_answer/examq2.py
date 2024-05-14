from scipy import optimize
from scipy import interpolate
from types import SimpleNamespace
import numpy as np


def utility(c, par):
    """ Intantaneous utlity of consumption c at parametrization par"""
    return c**(1-par.rho)/(1-par.rho)


def evaluate2(c, m, par):
    """ Evaluates utility at time 2 of given consumption and cash on hand.
    Inputs:
    c (float): Consumption at time 2
    m (float): Cash on hand at time 2
    par (SimpleNamespace): Parametrization 
    
    Returns:
    u (float): Utility at time 2
    """
    # consumption utility
    uc = utility(c, par)

    # bequest utility
    a = m - c
    ub = par.nu*utility(a+par.kappa, par)

    # total utility
    u = uc + ub

    return u


def v2(m, par):
    """ Value of cash on hand m at time 2
    Inputs:
    m (float): Cash on hand at time 2
    par (SimpleNamespace): Parametrization

    Returns:
    v2 (float): Value at time 2
    c_opt (float): Optimal comsumption at time 2
    """
    # Solve bounded optimization (consumption bounded between 0 and m)
    obj2 = lambda c: -evaluate2(c, m, par)
    res =  optimize.minimize_scalar(obj2, method='bounded', bounds=(0, m))
    assert res.success

    c_opt = res.x
    v2 = -res.fun
    return v2, c_opt


def evaluate1(c, m, s, par, interp_f,  tau = None):
    """ Evaluates utility at time 1 given consumption, cash on hand and study choice.
    Inputs:
    c (float): Consumption at time 1
    m (float): Cash on hand at time 1
    s (int, =0/1): Choice of whether to study or not
    par (SimpleNamespace): Parametrization
    interp_f (callable): Interpolater for value of cash on hand at time 2
    tau (float):  Cost to study. If tau = None, tau is set to par.tau.

    Returns:
    u (float): Expected utility at time 1
    """

    # If no value of tau is specified, tau is set equal to par.tau.
    if tau is None:
        tau = par.tau

    # Expected value of period 2
    a = m - c - tau*s
    Ev2 = par.p*interp_f([a*(1+par.r)+par.ybar + par.gamma*s + par.Delta])[0]
    Ev2 += (1-par.p)*interp_f([a*(1+par.r)+par.ybar + par.gamma*s - par.Delta])[0]

    # Utility of consumption at time 1
    uc = utility(c, par)

    # Total utility
    u = uc + par.beta*Ev2
    return u



def v1(m, par, interp_f, tau = None):
    """ Value of cash on hand at time 1
    Inputs:
    m (float): Cash on hand at time 1
    par (SimpleNamespace): Parametrization
    interp_f (callable): Interpolater for value of cash on hand at time 2
    tau (float):  Cost to study. If tau = None, tau is set to par.tau. 

    Returns:
    v1 (float): Value at time 1
    c_opt (float): Optimal consumption at time 1
    s_opt (int, =0/1): Optimal study choice
    """
    # If no value of tau is specified, tau is set equal to par.tau.
    if tau is None:
        tau = par.tau

    # solve with s = 1
    obj1s1 = lambda c: -evaluate1(c, m, s=1, par=par, interp_f = interp_f)
    as1 = m - tau
    res_study = optimize.minimize_scalar(obj1s1, method='bounded', bounds=(0, as1))
    assert res_study.success

    # solve with s = 0
    obj1s0 = lambda c: -evaluate1(c, m, s=0, par=par, interp_f = interp_f)
    res_not = optimize.minimize_scalar(obj1s0, method='bounded', bounds=(0,m))
    assert res_not.success

    # return optimal solution
    if res_study.fun <= res_not.fun:
        s_opt = 1
        c_opt = res_study.x
        v1 = -res_study.fun
    else:
        s_opt = 0
        c_opt = res_not.x
        v1 = -res_not.fun
    return v1, c_opt, s_opt


def study_gain(m, tau, interp_f, par):
    """ Expected utility gain from studying given cash on hand and cost of studying.
    Inputs:
    m (float): Cash on hand at time 1
    tau (float): Cost of studying
    interp_f (callable): Interpolater for value of cash on hand at time 2
    par (SimpleNamespace): Parametrization

    Returns:
    v1_gain (float): Expected utility gain from studying
    """
    # solve with s = 1
    obj1s1 = lambda c: -evaluate1(c, m, s=1, tau = tau, par=par, interp_f=interp_f)
    as1 = m - tau
    res_study = optimize.minimize_scalar(obj1s1, method='bounded', bounds=(0, as1))
    assert res_study.success

    # solve with s = 0
    obj1s0 = lambda c: -evaluate1(c, m, s=0, tau=tau, par=par, interp_f=interp_f)
    res_not = optimize.minimize_scalar(obj1s0, method='bounded', bounds=(0,m))
    assert res_not.success

    # return difference in utility
    v1_gain = (-res_study.fun) - (-res_not.fun)
    return v1_gain