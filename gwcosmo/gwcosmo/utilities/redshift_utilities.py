import numpy as np
import astropy.constants as const

speed_of_light = const.c.to('km/s').value

def sph2vec(ra, dec):
    return np.array([np.sin(np.pi/2.-dec)*np.cos(ra),np.sin(np.pi/2.-dec)*np.sin(ra),np.cos(np.pi/2.-dec)])

def zhelio_to_zcmb(ra, dec, z_helio):
    ra_cmb = 167.99
    dec_cmb = -7.22
    v_cmb = 369.
    z_gal_cmb = v_cmb*np.dot(sph2vec(ra_cmb,dec_cmb),sph2vec(ra,dec))/speed_of_light
    return (1.+z_helio)*(1.+z_gal_cmb)-1.