"""
Module with Schechter magnitude function:
(C) Walter Del Pozzo (2014)
"""
from numpy import *
from scipy.integrate import quad
import numpy as np


class SchechterMagFunctionInternal(object):
    def __init__(self, Mstar, alpha, phistar):
        self.Mstar = Mstar
        self.phistar = phistar
        self.alpha = alpha
        self.norm = None

    def evaluate(self, m):
        return 0.4*log(10.0)*self.phistar \
               * pow(10.0, -0.4*(self.alpha+1.0)*(m-self.Mstar)) \
               * exp(-pow(10, -0.4*(m-self.Mstar)))

    def normalise(self, mmin, mmax):
        if self.norm is None:
            self.norm = quad(self.evaluate, mmin, mmax)[0]

    def pdf(self, m):
        return self.evaluate(m)/self.norm


def SchechterMagFunction(H0=70., Mstar_obs=-19.70, alpha=-1.07, phistar=1.):
    """
    Returns a Schechter magnitude function fort a given set of parameters

    Parameters
    ----------
    H0 : Hubble parameter in km/s/Mpc (default=70.)
    Mstar_obs : observed characteristic magnitude used to define
                Mstar = Mstar_obs + 5.*np.log10(H0/100.) (default=-20.457)
    alpha : observed characteristic slope (default=-1.07)
    phistar : density (can be set to unity)

    Example usage
    -------------

    smf = SchechterMagFunction(H0=70., Mstar_obs=-20.457, alpha=-1.07)
    (integral, error) = scipy.integrate.quad(smf, Mmin, Mmax)
    """
    Mstar = Mstar_obs + 5.*np.log10(H0/100.)
    smf = SchechterMagFunctionInternal(Mstar, alpha, phistar)
    return smf.evaluate


def M_Mobs(H0, M_obs):
    """
    Given an observed absolute magnitude, returns absolute magnitude
    """
    return M_obs + 5.*np.log10(H0/100.)
