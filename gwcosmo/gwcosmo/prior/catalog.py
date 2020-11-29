"""
Module containing functionality for creation and management of galaxy catalogs.
Ignacio Magana
"""
import gwcosmo
import numpy as np
import healpy as hp
import pandas as pd

import h5py
import pkg_resources
import time
import progressbar

from ..utilities.standard_cosmology import *
from ..utilities.schechter_function import *
# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo',
                                                    'data/catalog_data/')

Kcorr_bands = {'B':'B', 'K':'K', 'u':'r', 'g':'r', 'r':'g', 'i':'g', 'z':'r'}
Kcorr_signs = {'B':1, 'K':1, 'u':1, 'g':1, 'r':-1, 'i':-1, 'z':-1}
color_names = {'B':None, 'K':None, 'u':'u - r', 'g':'g - r', 'r':'g - r', 'i':'g - i', 'z':'r - z'}
color_limits = {'u - r':[-0.1,2.9], 'g - r':[-0.1,1.9], 'g - i':[0,3], 'r - z':[0,1.5]}

def redshiftUncertainty(ra, dec, z, sigmaz, m, tempsky, luminosity_weights):
    """    
    A function which "smears" out galaxies in the catalog, therefore
    incorporating redshift uncetainties.
    """
    sort = np.argsort(m)
    ra, dec, z, sigmaz, m, tempsky = ra[sort], dec[sort], z[sort], sigmaz[sort], m[sort], tempsky[sort]
    
    if luminosity_weights is not 'constant':
        mlim = np.percentile(m,0.01) # more draws for galaxies in brightest 0.01 percent
    else:
        mlim = 1.0

    z_uncert = []
    ra_uncert = []
    dec_uncert = []
    m_uncert = []
    tempsky_uncert = []
    nsmear_uncert = []
    for k in range(len(m)):
        if m[k]<=mlim:
            nsmear = 10000
        else:
            nsmear = 20
        z_uncert.append(np.random.normal(z[k],sigmaz[k],nsmear))
        ra_uncert.append(np.repeat(ra[k],nsmear))
        dec_uncert.append(np.repeat(dec[k],nsmear))
        m_uncert.append(np.repeat(m[k],nsmear))
        tempsky_uncert.append(np.repeat(tempsky[k],nsmear))
        nsmear_uncert.append(np.repeat(nsmear,nsmear))

    z_uncert = np.concatenate(z_uncert).ravel()
    ra_uncert = np.concatenate(ra_uncert).ravel()
    dec_uncert = np.concatenate(dec_uncert).ravel()
    m_uncert = np.concatenate(m_uncert).ravel()
    tempsky_uncert = np.concatenate(tempsky_uncert).ravel()
    nsmear_uncert = np.concatenate(nsmear_uncert).ravel()

    return ra_uncert, dec_uncert, z_uncert, m_uncert, tempsky_uncert, nsmear_uncert

class galaxy(object):
    """
    Class to store galaxy objects.

    Parameters
    ----------
    index : galaxy index
    ra : Right ascension in radians
    dec : Declination in radians
    z : Galaxy redshift
    m : Apparent magnitude dictonary
    sigmaz : Galaxy redshift uncertainty
    """
    def __init__(self, index=0, ra=0, dec=0, z=1, m={'band':20}, sigmaz=0, Kcorr=0):
        self.index = index
        self.ra = ra
        self.dec = dec
        self.z = z
        self.m = m
        self.Kcorr = Kcorr
        self.sigmaz = sigmaz
        self.dl = self.luminosity_distance()
        self.M = self.absolute_magnitude()
        self.L = self.luminosity()

    def luminosity_distance(self, H0=70.):
        return dl_zH0(self.z, H0)
    
    def absolute_magnitude(self, H0=70., band=None):
        if band is not None:
            return M_mdl(self.m[band], self.dl)
        else:
            M = {}
            for band in self.m:
                M[band] = M_mdl(self.m[band], self.dl)
            return M

    def luminosity(self, band=None):
        if band is not None:
            return L_M(self.M[band])
        else:
            L = {}
            for band in self.M:
                L[band] = L_M(self.M[band])
            return L
 

class galaxyCatalog(object):
    """
    Galaxy catalog class stores a dictionary of galaxy objects.

    Parameters
    ----------
    catalog_file : Path to catalog.hdf5 file
    skymap_filename : Path to skymap.fits(.gz) file
    thresh: Probablity contained within sky map region
    band: key of band
    """
    def __init__(self, catalog_file=None, skymap_filename=None, thresh=.9, band='B', Kcorr=False):
        if catalog_file is not None:
            self.catalog_file = catalog_file
            self.dictionary = self.__load_catalog()
            self.extract_galaxies(skymap_filename=skymap_filename, thresh=thresh, band=band, Kcorr=Kcorr)

        if catalog_file is None:
            self.catalog_name = ""
            self.dictionary = {'ra': [], 'dec': [], 'z': [], 'sigmaz': [],
                               'skymap_indices': [], 'radec_lim': [], 'm_{0}'.format(band): []}

    def __load_catalog(self):
        return h5py.File(self.catalog_file,'r')

    def nGal(self):
        return len(self.dictionary['z'])

    def get_galaxy(self, index):
        return galaxy(index, self.ra[index], self.dec[index],
                      self.z[index], {self.band: self.m[index]},
                      self.sigmaz[index])

    def mth(self):
        m = self.m
        if sum(m) == 0:
            mth = 25.0
        else:
            mth = np.median(m)
        return mth

    def extract_galaxies(self, skymap_filename=None, thresh=.9, band='B', Kcorr=False):
        band_key = 'm_{0}'.format(band)
        if Kcorr is True:
            band_Kcorr = Kcorr_bands[band]
            band_Kcorr_key = 'm_{0}'.format(band_Kcorr)
            self.color_name = color_names[band]
            self.color_limit = color_limits[self.color_name]
        if skymap_filename:
            skymap = hp.read_map(skymap_filename, verbose=False)
        gal_ind = self.dictionary['skymap_indices'][:]
        if skymap_filename is None:
            ra = self.dictionary['ra'][:]
            dec = self.dictionary['dec'][:]
            z = self.dictionary['z'][:]
            sigmaz = self.dictionary['sigmaz'][:]
            m = self.dictionary[band_key][:]
            if Kcorr is True:
                m_K = self.dictionary[band_Kcorr_key][:]
            radec_lim = self.dictionary['radec_lim'][:]
        else:
            ind = self.galaxies_within_region(skymap, gal_ind, thresh)
            ra = self.dictionary['ra'][ind]
            dec = self.dictionary['dec'][ind]
            z = self.dictionary['z'][ind]
            sigmaz = self.dictionary['sigmaz'][ind]
            m = self.dictionary[band_key][ind]
            if Kcorr is True:
                m_K = self.dictionary[band_Kcorr_key][ind]
            radec_lim = self.dictionary['radec_lim'][:]
        self.band = band
        self.radec_lim = radec_lim
        if Kcorr is True:
            mask = (~np.isnan(m))&(~np.isnan(m_K))
            m_K = m_K[mask]
        else:
            mask = ~np.isnan(m)
        ra, dec, z, m, sigmaz = ra[mask], dec[mask], z[mask], m[mask], sigmaz[mask]
        self.ra, self.dec, self.z, self.sigmaz, self.m = ra, dec, z, sigmaz, m
        if Kcorr is True:
            self.color = Kcorr_signs[band]*(m - m_K)
        else:
            self.color = np.zeros(len(m))
        return ra, dec, z, m, sigmaz
    
    def above_percentile(self, skymap, thresh):
        """Returns indices of array within the given threshold
        credible region."""
        #  Sort indicies of sky map
        ind_sorted = np.argsort(-skymap)
        #  Cumulatively sum the sky map
        cumsum = np.cumsum(skymap[ind_sorted])
        #  Find indicies contained within threshold area
        lim_ind = np.where(cumsum > thresh)[0][0]
        return ind_sorted[:lim_ind]

    def galaxies_within_region(self, skymap, gal_ind, thresh):
        """Returns boolean array of whether galaxies are within
        the sky map's credible region above the given threshold"""
        skymap_ind = self.above_percentile(skymap, thresh)
        return np.in1d(gal_ind, skymap_ind)

