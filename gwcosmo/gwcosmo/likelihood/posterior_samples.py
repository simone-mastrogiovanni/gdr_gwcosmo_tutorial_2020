"""
LALinference posterior samples class and methods
Ignacio Magana, Ankan Sur
"""
import numpy as np
import healpy as hp
from scipy.stats import gaussian_kde
from scipy import integrate, interpolate, random
from astropy import units as u
from astropy import constants as const
from astropy.table import Table
import h5py
from bilby.core.prior import Uniform, PowerLaw, PriorDict, Constraint
from bilby import gw
from ..utilities.standard_cosmology import z_dlH0, fast_cosmology
from ..prior.priors import *

from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import astropy.constants as constants
from scipy.interpolate import interp1d

Om0 = 0.308
zmin = 0.0001
zmax = 10
zs = np.linspace(zmin, zmax, 10000)
cosmo = fast_cosmology(Omega_m=Om0)

def constrain_m1m2(parameters):
    converted_parameters = parameters.copy()
    converted_parameters['m1m2'] = parameters['mass_1'] - parameters['mass_2']
    return converted_parameters


class posterior_samples(object):
    """
    Posterior samples class and methods.

    Parameters
    ----------
    posterior_samples : Path to posterior samples file to be loaded.
    """
    def __init__(self, posterior_samples=None):
        self.posterior_samples = posterior_samples
        try:
            self.load_posterior_samples()
        except:
            print("No posterior samples were specified")


    def E(self,z,Om):
        return np.sqrt(Om*(1+z)**3 + (1.0-Om))

    def dL_by_z_H0(self,z,H0,Om0):
        speed_of_light = constants.c.to('km/s').value
        cosmo = fast_cosmology(Omega_m=Om0)
        return cosmo.dl_zH0(z, H0)/(1+z) + speed_of_light*(1+z)/(H0*self.E(z,Om0))

    def jacobian_times_prior(self,z,H0,Om0=0.308):

        cosmo = fast_cosmology(Omega_m=Om0)
        jacobian = np.power(1+z,2)*self.dL_by_z_H0(z,H0,Om0)
        dl = cosmo.dl_zH0(z, H0)
        return jacobian*(dl**2)

    def load_posterior_samples(self):
        """
        Method to handle different types of posterior samples file formats.
        Currently it supports .dat (LALinference), .hdf5 (GWTC-1),
        .h5 (PESummary) and .hdf (pycbcinference) formats.
        """
        if self.posterior_samples[-3:] == 'dat':
            samples = np.genfromtxt(self.posterior_samples, names=True)
            try:
                self.distance = samples['dist']
            except KeyError:
                try:
                    self.distance = samples['distance']
                except KeyError:
                    print("No distance samples found.")
            self.ra = samples['ra']
            self.dec = samples['dec']
            self.mass_1 = samples['mass_1']
            self.mass_2 = samples['mass_2']
            self.nsamples = len(self.distance)

        if self.posterior_samples[-4:] == 'hdf5':
            if self.posterior_samples[-11:] == 'GWTC-1.hdf5':
                if self.posterior_samples[-20:] == 'GW170817_GWTC-1.hdf5':
                    dataset_name = 'IMRPhenomPv2NRT_lowSpin_posterior'
                else:
                    dataset_name = 'IMRPhenomPv2_posterior'
                file = h5py.File(self.posterior_samples, 'r')
                data = file[dataset_name]
                self.distance = data['luminosity_distance_Mpc']
                self.ra = data['right_ascension']
                self.dec = data['declination']
                self.mass_1 = data['m1_detector_frame_Msun']
                self.mass_2 = data['m2_detector_frame_Msun']
                self.nsamples = len(self.distance)
                file.close()

        if self.posterior_samples[-2:] == 'h5':
            file = h5py.File(self.posterior_samples, 'r')
            approximants = ['C01:PhenomPNRT-HS', 'C01:NRSur7dq4',
                            'C01:IMRPhenomPv3HM', 'C01:IMRPhenomPv2',
                            'C01:IMRPhenomD']
            for approximant in approximants:
                try:
                    data = file[approximant]
                    print("Using "+approximant+" posterior")
                    break
                except KeyError:
                    continue

            self.distance = data['posterior_samples']['luminosity_distance']
            self.ra = data['posterior_samples']['ra']
            self.dec = data['posterior_samples']['dec']
            self.mass_1 = data['posterior_samples']['mass_1']
            self.mass_2 = data['posterior_samples']['mass_2']
            self.nsamples = len(self.distance)
            file.close()

        if self.posterior_samples[-3:] == 'hdf':
            file = h5py.File(self.posterior_samples, 'r')
            self.distance = file['samples/distance'][:]
            self.ra = file['samples/ra'][:]
            self.dec = file['samples/dec'][:]
            self.mass_1 = file['samples/mass_1'][:]
            self.mass_2 = file['samples/mass_2'][:]
            self.nsamples = len(self.distance)
            file.close()

    def marginalized_sky(self):
        """
        Computes the marginalized sky localization posterior KDE.
        """
        return gaussian_kde(np.vstack((self.ra, self.dec)))

    def compute_source_frame_samples(self, H0):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        dLs = cosmo.luminosity_distance(zs).to(u.Mpc).value
        z_at_dL = interp1d(dLs,zs)
        redshift = z_at_dL(self.distance)
        mass_1_source = self.mass_1/(1+redshift)
        mass_2_source = self.mass_2/(1+redshift)
        return redshift, mass_1_source, mass_2_source

    def reweight_samples(self, H0, name, alpha=1.6, mmin=5, mmax=100, seed=1):
        # Prior distribution used in the LVC analysis
        prior = distance_distribution(name=name)
        # Prior distribution used in this work
        new_prior = mass_distribution(name=name, alpha=alpha, mmin=mmin, mmax=mmax)

        # Get source frame masses
        redshift, mass_1_source, mass_2_source = self.compute_source_frame_samples(H0)

        # Re-weight
        weights = new_prior.joint_prob(mass_1_source,mass_2_source)/ prior.prob(self.distance)
        np.random.seed(seed)
        draws = np.random.uniform(0, max(weights), weights.shape)
        keep = weights > draws
        m1det = self.mass_1[keep]
        m2det = self.mass_2[keep]
        dl = self.distance[keep]
        return dl, weights


    def marginalized_redshift_reweight(self, H0, name, alpha=1.6, mmin=5, mmax=100):
        """
        Computes the marginalized distance posterior KDE.
        """
        # Prior distribution used in this work
        new_prior = mass_distribution(name=name, alpha=alpha, mmin=mmin, mmax=mmax)

        # Get source frame masses
        redshift, mass_1_source, mass_2_source = self.compute_source_frame_samples(H0)

        # Re-weight
        weights = new_prior.joint_prob(mass_1_source,mass_2_source)/self.jacobian_times_prior(redshift,H0)
        norm = np.sum(weights)
        return gaussian_kde(redshift,weights=weights), norm

    def marginalized_redshift(self, H0):
        """
        Computes the marginalized distance posterior KDE.
        """
        # Get source frame masses
        redshift, mass_1_source, mass_2_source = self.compute_source_frame_samples(H0)

        # remove dl^2 prior and include dz/ddL jacobian
        weights = 1/(self.dL_by_z_H0(redshift,H0,Om0)*cosmo.dl_zH0(redshift,H0)**2)
        norm = np.sum(weights)
        return gaussian_kde(redshift,weights=weights), norm

    def marginalized_distance_reweight(self, H0, name, alpha=1.6, mmin=5, mmax=100, seed=1):
        """
        Computes the marginalized distance posterior KDE.
        """
        dl, weights = self.reweight_samples(H0, name, alpha=alpha, mmin=mmin, mmax=mmax, seed=seed)
        norm = np.sum(weights)/len(weights)
        return gaussian_kde(dl), norm

    def marginalized_distance(self, H0):
        """
        Computes the marginalized distance posterior KDE.
        """
        norm = 1
        return gaussian_kde(self.distance), norm
