"""
Module to compute and handle skymaps
Rachel Gray, Ignacio Magana, Archisman Ghosh, Ankan Sur
"""
import numpy as np
import scipy.stats
from astropy.io import fits
import healpy as hp
from scipy import interpolate
from scipy.stats import norm
import sys


# RA and dec from HEALPix index
def ra_dec_from_ipix(nside, ipix, nest=False):
    (theta, phi) = hp.pix2ang(nside, ipix, nest=nest)
    return (phi, np.pi/2.-theta)


# HEALPix index from RA and dec
def ipix_from_ra_dec(nside, ra, dec, nest=False):
    (theta, phi) = (np.pi/2.-dec, ra)
    return hp.ang2pix(nside, theta, phi, nest=nest)


class skymap(object):
    """
    Read a FITS file and return interpolation kernels on the sky.
    TODO: Rework to use ligo.skymap
    """
    def __init__(self, filename):
        """
        Input parameters:
        - filename : FITS file to load from
        """
        try:
            prob, header = hp.read_map(filename, field=[0, 1, 2, 3],
                                       h=True, nest=True)
            self.prob = prob[0]
            self.distmu = prob[1]
            self.distsigma = prob[2]
            self.distnorm = prob[3]
        except IndexError:
            self.prob = hp.read_map(filename, nest=True)
            self.distmu = np.ones(len(self.prob))
            self.distsigma = np.ones(len(self.prob))
            self.distnorm = np.ones(len(self.prob))

        self.nested = True
        self.npix = len(self.prob)
        self.nside = hp.npix2nside(self.npix)
        colat, self.ra = hp.pix2ang(self.nside, range(len(self.prob)),
                                    nest=self.nested)
        self.dec = np.pi/2.0 - colat

    def probability(self, ra, dec, dist):
        """
        returns probability density at given ra, dec, dist
        p(ra,dec) * p(dist | ra,dec )
        RA, dec : radians
        dist : Mpc
        """
        theta = np.pi/2.0 - dec
        # Step 1: find 4 nearest pixels
        (pixnums, weights) = \
            hp.get_interp_weights(self.nside, theta, ra,
                                  nest=self.nested, lonlat=False)

        dist_pdfs = [scipy.stats.norm(loc=self.mean[i], scale=self.sigma[i])
                     for i in pixnums]
        # Step 2: compute p(ra,dec)
        # p(ra, dec) = sum_i weight_i p(pixel_i)
        probvals = np.array([self.distnorm[i] * dist_pdfs[i].pdf(dist)
                            for i, pixel in enumerate(pixnums)])
        skyprob = self.prob[pixnums]
        p_ra_dec = np.sum(weights * probvals * skyprob)

        return(p_ra_dec)

    def skyprob(self, ra, dec):
        """
        Return the probability of a given sky location
        ra, dec: radians
        """
        ipix_gal = hp.ang2pix(self.nside, np.pi/2.0-dec, ra, nest=self.nested)
        return self.prob[ipix_gal]

    def marginalized_distance(self):
        mu = self.distmu[(self.distmu<np.inf) & (self.distmu>0)]
        distmin = 0.5*min(mu)
        distmax = 2*max(mu)
        dl = np.linspace(distmin, distmax, 200)
        dp_dr = [np.sum(self.prob * r**2 * self.distnorm *
                        norm(self.distmu, self.distsigma).pdf(r)) for r in dl]
        return dl, dp_dr

    def lineofsight_distance(self, ra, dec):
        ipix = ipix_from_ra_dec(self.nside, ra, dec, nest=self.nested)
        mu = self.distmu[(self.distmu<np.inf) & (self.distmu>0)]
        distmin = 0.5*min(mu)
        distmax = 2*max(mu)
        r = np.linspace(distmin, distmax, 200)
        dp_dr = r**2 * self.distnorm[ipix] * norm(self.distmu[ipix],
                                                  self.distsigma[ipix]).pdf(r)
        return r, dp_dr

    def probability(self, ra, dec, dist):
        """
        returns probability density at given ra, dec, dist
        p(ra,dec) * p(dist | ra,dec )
        RA, dec : radians
        dist : Mpc
        """
        theta = np.pi/2.0 - dec
        # Step 1: find 4 nearest pixels
        (pixnums, weights) = hp.get_interp_weights(self.nside,
                                                   theta, ra, nest=self.nested,
                                                   lonlat=False)

        dist_pdfs = [norm(loc=self.distmu[i], scale=self.distsigma[i])
                     for i in pixnums]
        # Step 2: compute p(ra,dec)
        # p(ra, dec) = sum_i weight_i p(pixel_i)
        probvals = np.array([self.distnorm[i] * dist_pdfs[i].pdf(dist)
                            for i, pixel in enumerate(pixnums)])
        skyprob = self.prob[pixnums]
        p_ra_dec = np.sum(weights * probvals * skyprob)

        return(p_ra_dec)
