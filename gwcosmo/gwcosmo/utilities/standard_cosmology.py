"""
Module for basic late-time cosmology calculations.
Currently implements only a flat universe.

Constants
---------
c : speed of light in km/s
Omega_m : matter fraction
H0 : Hubble parameter

(c) Archisman Ghosh, 2013-Nov
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import splrep, splev
import lal

c = lal.C_SI/1000.  # 2.99792458e+05 # in km/s
Omega_m = 0.3  # 0.3175 # PLANCK best fit
H0 = 70  # 67.11 # in km/s/Mpc


def h(z, Omega_m=Omega_m):
    """
    Returns dimensionless redshift-dependent hubble parameter.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction

    Returns
    -------
    dimensionless h(z) = 1/sqrt(Omega_m*(1+z)^3 + Omega_Lambda)
    """
    Omega_Lambda = (1-Omega_m)
    return np.sqrt(Omega_m*(1+z)**3 + Omega_Lambda)


def dcH0overc(z, Omega_m=Omega_m):
    """
    Returns dimensionless combination dc*H0/c
    given redshift and matter fraction.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction

    Returns
    -------
    dimensionless combination dc*H0/c = \int_0^z dz'/h(z')
    """
    Omega_Lambda = (1-Omega_m)
    integrand = lambda zz: 1./np.sqrt(Omega_m*(1+zz)**3 + Omega_Lambda)
    return integrate.quad(integrand, 0, z)[0]  # in km/s


def dLH0overc(z, Omega_m=Omega_m):
    """
    Returns dimensionless combination dL*H0/c
    given redshift and matter fraction.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction

    Returns
    -------
    dimensionless combination dL*H0/c = (1+z) * \int_0^z dz'/h(z')
    """
    return (1+z)*dcH0overc(z, Omega_m)


def volume_z(z, Omega_m=Omega_m):
    """
    Returns the cosmological volume at the given redshift.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction

    Returns
    -------
    volume element (\int_0^z dz'/h(z'))^2 / h(z): dimensionless
    """
    return dcH0overc(z, Omega_m)**2/h(z, Omega_m)


def volume_time_z(z, Omega_m=Omega_m):
    """
    Returns the cosmological volume time element at a given redshift.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction

    Returns
    -------
    volume time element (\int_0^z dz'/h(z'))^2 / (1+z)h(z)
    """
    return volume_z(z, Omega_m=Omega_m)/(1.0+z)


def prefactor_volume_dHoc(dHoc, Omega_m=Omega_m, tolerance_z=1e-06, z=None):
    """
    Returns the prefactor modifying dL^2*ddL
    for the cosmological volume element.

    Parameters
    ----------
    dLH0overc : dimensionless combination dL*H0/c
    Omega_m : matter fraction
    tolerance_z : (optional) tolerated error in redshift. default = 1e-06
    z : (optional) redshift, if it has been calculated already

    Returns
    -------
    prefactor, (1+z)^(-3) * (1 - 1 / (1 + (1+z)^2/(dHoc*h(z))))
    """
    if z is None:
        z = redshift(dHoc, Omega_m, tolerance_z)
    return (1+z)**(-3.) * (1 - 1. / (1 + (1+z)**2/(dHoc*h(z, Omega_m))))


def volume_dHoc(dHoc, Omega_m=Omega_m, tolerance_z=1e-06, z=None):
    """
    Returns cosmological volume at the given dL*H0/c.

    Parameters
    ----------
    dLH0overc : dimensionless combination dL*H0/c
    Omega_m : matter fraction
    tolerance_z : (optional) tolerated error in redshift. default = 1e-06
    z : (optional) redshift, if it has been calculated already

    Returns
    -------
    volume, dHoc^2 * (1+z)^(-3) * (1 - 1 / (1 + (1+z)^2/(dHoc*h(z))))
    """
    return dHoc**2*prefactor_volume_dHoc(dHoc, Omega_m, tolerance_z, z=z)


def redshift(dHoc, Omega_m=Omega_m, tolerance_z=1e-06):
    """
    Returns redshift given dimensionless combination dL*H0/c
    and matter fraction.

    Parameters
    ----------
    dLH0overc : dimensionless combination dL*H0/c
    Omega_m : matter fraction
    tolerance_z : (optional) tolerated error in redshift. default = 1e-06.

    Returns
    -------
    redshift, z
    """
    min_z = 0.
    max_z = 1.
    error_z = max_z-min_z
    while error_z > tolerance_z:
        if dLH0overc(max_z, Omega_m) < dHoc:
            min_z = max_z
            max_z *= 2
        elif dLH0overc((max_z+min_z)/2., Omega_m) < dHoc:
            min_z = (max_z+min_z)/2.
        else:
            max_z = (max_z+min_z)/2.
        error_z = max_z-min_z
    return (max_z+min_z)/2.


# Distance modulus given luminosity distance
def DistanceModulus(dL):
    """
    Returns distance modulus given luminosity distance

    Parameters
    ----------
    dL : luminosity distance in Mpc

    Returns
    -------
    distance modulus = 5*np.log10(dL)+25
    """
    return 5*np.log10(dL)+25  # dL has to be in Mpc


def dl_mM(m, M, Kcorr=0.):
    """
    returns luminosity distance in Mpc given
    apparent magnitude and absolute magnitude
    
    Parameters
    ----------
    m : apparent magnitude
    M : absolute magnitude in the source frame
    Kcorr : (optional) K correction, to convert absolute magnitude from the 
        observed band to the source frame band (default=0).  If fluxes are 
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.
        
    Returns
    -------
    Luminosity distance in Mpc
    """
    return 10**(0.2*(m-M-Kcorr-25))


def L_M(M):
    """
    Returns luminosity when given an absolute magnitude
    
    Parameters
    ----------
    M : absolute magnitude in the source frame
    
    Returns
    -------
    Luminosity in Watts
    """
    # TODO: double check use of L0=3.0128e28
    return 3.0128e28*10**(-0.4*M)


def M_mdl(m, dl, Kcorr=0.):
    """
    Returns a source's absolute magnitude given
    apparent magnitude and luminosity distance
    If a K correction is supplied it will be applied
    
    Parameters
    ----------
    m : apparent magnitude
    dl : luminosity distance in Mpc
    Kcorr : (optional) K correction, to convert absolute magnitude from the 
        observed band to the source frame band (default=0).  If fluxes are 
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.
    
    Returns
    -------
    Absolute magnitude in the source frame
    """
    return m - DistanceModulus(dl) - Kcorr


def L_mdl(m, dl, Kcorr=0.):
    """
    Returns luminosity when given apparent magnitude and luminosity distance
    If a K correction is supplied it will be applied
    
    Parameters
    ----------
    m : apparent magnitude
    dl : luminosity distance in Mpc
    Kcorr : (optional) K correction, to convert absolute magnitude from the 
        observed band to the source frame band (default=0).  If fluxes are 
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.
    
    Returns
    -------
    Luminosity in the source frame
    """
    return L_M(M_mdl(m, dl, Kcorr=Kcorr))


# Rachel: I've put dl_zH0 and z_dlH0 in as place holders.
def dl_zH0(z, H0=70., Omega_m=0.3, linear=False):
    """
    Returns luminosity distance given distance and cosmological parameters

    Parameters
    ----------
    z : redshift
    H0 : Hubble parameter in km/s/Mpc (default=70.)
    Omega_m : matter fraction (default=0.3)
    linear : assumes local cosmology and suppresses
    non-linear effects (default=False)

    Returns
    -------
    luminosity distance, dl (in Mpc)
    """
    if linear:
        # Local cosmology
        if z == 0:
            return 10**(-30)
        else:
            return z*c/H0
    else:
        # Standard cosmology
        return dLH0overc(z, Omega_m=Omega_m)*c/H0


def z_dlH0(dl, H0=70., Omega_m=0.3, linear=False):
    """
    Returns redshift given luminosity distance and cosmological parameters

    Parameters
    ----------
    dl : luminosity distance in Mpc
    H0 : Hubble parameter in km/s/Mpc (default=70.)
    Omega_m : matter fraction (default=0.3)
    linear : assumes local cosmology and suppresses
    non-linear effects (default=False)

    Returns
    -------
    redshift, z
    """
    if linear:
        # Local cosmology
        return dl*H0/c
    else:
        # Standard cosmology
        return redshift(dl*H0/c, Omega_m=Omega_m)


class redshift_prior(object):
    """
    p(z|Omega_m)
    """
    def __init__(self, Omega_m=0.3, zmax=10.0, linear=False):
        self.Omega_m = Omega_m
        self.linear = linear
        self.zmax = zmax
        z_array = np.linspace(0.0, self.zmax, 5000)
        lookup = np.array([volume_time_z(z, Omega_m=self.Omega_m)
                          for z in z_array])
        self.interp = splrep(z_array, lookup)

    def p_z(self, z):
        return splev(z, self.interp, ext=3)

    def __call__(self, z):
        if self.linear:
            return z*z
        else:
            return self.p_z(z)


class fast_cosmology(object):
    """
    Precompute things which rely on choice of Omega_m
    in order to speed things up

    Parameters
    ----------
    Omega_m : matter fraction (default=0.3)
    zmax : upper limit for redshift (default=4.0)
    linear : assumes local cosmology and suppresses
    non-linear effects (default=False)

    """
    def __init__(self, Omega_m=0.3, zmax=10.0, linear=False):
        self.Omega_m = Omega_m
        self.linear = linear
        self.zmax = zmax
        z_array = np.linspace(0.0, self.zmax, 5000)
        lookup = np.array([dLH0overc(z, Omega_m=self.Omega_m)
                          for z in z_array])
        self.interp = splrep(z_array, lookup)

    def dl_zH0(self, z, H0):
        """
        Returns luminosity distance given distance and cosmological parameters

        Parameters
        ----------
        z : redshift
        H0 : Hubble parameter in km/s/Mpc

        Returns
        -------
        luminosity distance, dl (in Mpc)
        """
        if self.linear:
            # Local cosmology
            return z*c/H0
        else:
            # Standard cosmology     
            return splev(z, self.interp, ext=3)*c/H0
