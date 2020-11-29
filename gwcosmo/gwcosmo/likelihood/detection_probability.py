"""
Detection probability
Rachel Gray, John Veitch, Ignacio Magana, Dominika Zieba
"""
import lal
from lal import ComputeDetAMResponse
import lalsimulation as lalsim
import numpy as np
from scipy.interpolate import interp1d, splev, splrep, interp2d
from scipy.integrate import quad
from scipy.stats import ncx2
from scipy.special import logit, expit
import healpy as hp
from gwcosmo.utilities.standard_cosmology import *
from gwcosmo.prior.priors import mass_sampling as mass_prior

import pickle
import time
import progressbar
import pkg_resources
import os


class DetectionProbability(object):
    """
    Class to compute the detection probability p(D |z, H0) as a function of z and H0
    by marginalising over masses, inclination, polarisation, and sky location for a 
    set of detectors at a given sensitivity.

    Parameters
    ----------
    mass_distribution : str
        choice of mass distribution ('BNS', 'NSBH', 'BBH-powerlaw', 'BBH-constant')
    psd : str
        Select between 'O1', 'O2', 'O3', 'O4low', 'O4high', 'O5' or the 'MDC' PSDs.
    detectors : list of str, optional
        list of detector names (default=['H1','L1'])
        Select from 'L1', 'H1', 'V1', 'K1'.
    Nsamps : int, optional
        Number of samples for monte carlo integration (default=5000)
    H0 : float or array, optional
        Value(s) of H0 at which to compute Pdet. If constant_H0 is True (default=70)
    network_snr_theshold : float, optional
        snr threshold for an individual detector (default=12)
    Omega_m : float, optional
        matter fraction of the universe (default=0.3)
    linear : bool, optional
        if True, use linear cosmology (default=False)
    basic : bool, optional
        if True, don't redshift masses (for use with the MDC) (default=False)
    alpha : float, optional
        slope of the power law p(m) = m^-\alpha where alpha > 0 (default=1.6)
    Mmin : float, optional
        specify minimum source frame mass for BBH-powerlaw distribution (default=5)
    Mmax : float, optional
        specify maximum source frame mass for BBH-powerlaw distribution  (default=50)
    M1, M2 : float, optional
        specify source masses in solar masses if using BBH-constant mass distribution (default=50,50)
    constant_H0 : bool, optional
        if True, set Hubble constant to 70 kms-1Mpc-1 for all calculations (default=False)
    full_waveform: bool, optional
        if True, use LALsimulation simulated inspiral waveform, otherwise use just the inspiral (default=True)
    
    """
    def __init__(self, mass_distribution, asd, detectors=['H1', 'L1'],
                 Nsamps=5000, H0=70, network_snr_threshold=12, Omega_m=0.308,
                 linear=False, basic=False, alpha=1.6, Mmin=5., Mmax=50., M1=50., M2=50.,
                 constant_H0=False, full_waveform=True, seed=1000):
        self.data_path = pkg_resources.resource_filename('gwcosmo', 'data/')
        self.mass_distribution = mass_distribution
        self.asd = asd
        self.detectors = detectors
        self.Nsamps = Nsamps
        self.H0vec = H0
        self.snr_threshold = network_snr_threshold
        self.Omega_m = Omega_m
        self.linear = linear
        self.alpha = alpha
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.M1=M1
        self.M2=M2
        self.full_waveform = full_waveform
        self.constant_H0 = constant_H0
        self.seed = seed
       
        np.random.seed(seed)
        
        ASD_data = {}
        self.asds = {}    #this is now a dictionary of functions
        for det in self.detectors:
            if self.asd == 'MDC':
                ASD = np.genfromtxt(self.data_path + 'PSD_L1_H1_mid.txt')
                self.asds[det] = interp1d(ASD[:, 0], ASD[:, 1])
            else:
                ASD_data[det] = np.genfromtxt(self.data_path + det + '_'+ self.asd + '_strain.txt')
                self.asds[det] = interp1d(ASD_data[det][:, 0], ASD_data[det][:, 1])

        self.cosmo = fast_cosmology(Omega_m=self.Omega_m, linear=self.linear)
        
        if self.full_waveform is True:
            self.z_array = np.logspace(-4.0, 1., 500)
        else:
            # TODO: For higher values of z (z=10) this goes
            # outside the range of the psds and gives an error
            self.z_array = np.logspace(-4.0, 0.5, 500)

        # set up the samples for monte carlo integral
        N = self.Nsamps
        self.RAs = np.random.rand(N)*2.0*np.pi
        r = np.random.rand(N)
        self.Decs = np.arcsin(2.0*r - 1.0)
        q = np.random.rand(N)
        self.incs = np.arccos(2.0*q - 1.0)
        self.psis = np.random.rand(N)*2.0*np.pi
        self.phis = np.random.rand(N)*2.0*np.pi
        if self.mass_distribution == 'BNS':
            mass_priors = mass_prior(self.mass_distribution)
            self.dl_array = np.linspace(1.0e-100, 1000.0, 500)
        if self.mass_distribution == 'NSBH':
            mass_priors = mass_prior(self.mass_distribution, self.alpha, self.Mmin, self.Mmax)
            self.dl_array = np.linspace(1.0e-100, 1000.0, 500)
        if self.mass_distribution == 'BBH-powerlaw':
            mass_priors = mass_prior(self.mass_distribution, self.alpha, self.Mmin, self.Mmax)
            self.dl_array = np.linspace(1.0e-100, 15000.0, 500)
        if self.mass_distribution == 'BBH-constant':
            mass_priors = mass_prior(self.mass_distribution, self.M1, self.M2)
            self.dl_array = np.linspace(1.0e-100, 15000.0, 500)
        m1, m2 = mass_priors.sample(N)
        self.m1 = m1*1.988e30
        self.m2 = m2*1.988e30

        self.M_min = np.min(self.m1)+np.min(self.m2)
        
        self.df = 1          #set sampling frequency interval to 1 Hz
        self.f_min = 10      #10 Hz minimum frequency
        self.f_max = 4999    #5000 Hz maximum frequency
        
        self.__interpolnum = {}    #this is now a dictionary of functions, one per detector
        for det in self.detectors:
            self.__interpolnum[det] = self.__numfmax_fmax(self.M_min, det)

        print("Calculating pdet with " + self.asd + " sensitivity and " +
              self.mass_distribution + " mass distribution.")
        
        if basic is True:
            self.interp_average_basic = self.__pD_dl_basic()
        
        elif constant_H0 is True:  
            self.prob = self.__pD_zH0(H0)
            logit_prob=logit(self.prob)
            logit_prob=np.where(logit_prob==float('+inf'), 100, logit_prob)   
            self.interp_average = interp1d(self.z_array, logit_prob, kind='cubic')
            
        else:
            self.prob = self.__pD_zH0_array(self.H0vec)
            
            #interpolation of prob is done in logit(prob)=prob/(1-prob) 
            #this prevents values from going above 1 and below 0
            #if prob=1, logit(prob)=inf. 
            #to solve this for interpolation purposes, set logit(prob=1)=100, so then expit(100)=logit^-1(100)=1
            #instead of 100 anything from 35 to sys.float_info.max can be set as in this range expit is effectively 1
            #yet: higher values make interpolation less effective
            
            logit_prob=logit(self.prob)
            for i in range (len(logit_prob)):
                logit_prob[i]=np.where(logit_prob[i]==float('+inf'), 100, logit_prob[i])   
            self.interp_average = interp2d(self.z_array, self.H0vec, logit_prob, kind='cubic')

    def mchirp(self, m1, m2):
        """
        Calculates the source chirp mass

        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg

        Returns
        -------
        Source chirp mass in kg
        """
        return np.power(m1*m2, 3.0/5.0)/np.power(m1+m2, 1.0/5.0)
    
    def mchirp_obs(self, m1, m2, z=0):
        """
        Calculates the redshifted chirp mass from source masses and a redshift

        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg
        z : float
            redshift (default=0)

        Returns
        -------
        float
            Redshifted chirp mass in kg
        """
        return (1+z)*self.mchirp(m1, m2)

    def __mtot(self, m1, m2):
        """
        Calculates the total source mass of the system

        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg

        Returns
        -------
        float
            Source total mass in kg
        """
        return m1+m2

    def __mtot_obs(self, m1, m2, z=0):
        """
        Calculates the total observed mass of the system

        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg
        z : float
            redshift (default=0)

        Returns
        -------
        float
            Observed total mass in kg
        """
        return (m1+m2)*(1+z)
    
    
    def __Fplus(self, detector, RA, Dec, psi, gmst):
        """
        Computes the 'plus' antenna pattern

        Parameters
        ----------
        detector : str
            name of detector in network (eg 'H1', 'L1')
        RA,Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds

        Returns
        -------
        float
            F_+ antenna response
        """
        detector = lalsim.DetectorPrefixToLALDetector(detector)
        return lal.ComputeDetAMResponse(detector.response, RA,
                                        Dec, psi, gmst)[0]

    def __Fcross(self, detector, RA, Dec, psi, gmst):
        """
        Computes the 'plus' antenna pattern

        Parameters
        ----------
        detector : str
            name of detector in network (eg 'H1', 'L1')
        RA,Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds

        Returns
        -------
        float
            F_x antenna response
        """
        detector = lalsim.DetectorPrefixToLALDetector(detector)
        return lal.ComputeDetAMResponse(detector.response, RA,
                                        Dec, psi, gmst)[1]
    
    def simulate_waveform(self, m1, m2, dl, inc, phi,
                          S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0., lAN=0., e=0., Ano=0.):       
        """
        Simulates frequency domain inspiral waveform

        Parameters
        ----------
        m1, m2 : float
            observed source masses in kg 
        dl : float
            luminosity distance in Mpc
        inc: float
            source inclination in radians
        phi : float
            source reference phase in radians
        S1x, S1y, S1z : float, optional
            x,y,z-components of dimensionless spins of body 1 (default=0.)
        S2x, S2y, S2z : float, optional
            x,y,z-components of dimensionless spin of body 2 (default=0.)
        lAN: float, optional
            longitude of ascending nodes (default=0.)
        e: float, optional
            eccentricity at reference epoch (default=0.)
        Ano: float, optional
            mean anomaly at reference epoch (default=0.)
        
        Returns
        -------
        lists
            hp and hc
        """  
                      
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
                    m1, m2,
                    S1x, S1y, S1z, S2x, S2y, S2z,                  
                    dl*1e6*lal.PC_SI, inc, phi, lAN, e, Ano,
                    self.df, self.f_min, self.f_max, 20,
                    lal.CreateDict(),
                    lalsim.IMRPhenomD)
        hp = hp.data.data     
        hc = hc.data.data
                          
        return hp,hc

    def simulate_waveform_response(self, hp, hc, RA, Dec, psi, gmst, detector):       
        """
        Applies antenna response to frequency domain inspiral waveform

        Parameters
        ----------
        hp, hc : lists
            plus and cross components of the frequency domain inspiral waveform
        RA, Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        phi : float
            source reference phase in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        detector : str
            name of detector in network (eg 'H1', 'L1')
        
        Returns
        -------
        complex array
            complex frequency series - detected h(f)
        
        real array
            array of frequencies corresponding to h(f)
        """  
        
        #apply antenna response 
        hf = hp*self.__Fplus(detector, RA, Dec, psi, gmst) + hc*self.__Fcross(detector, RA, Dec, psi, gmst)
        
        #recreate frequency array
        f_array=self.df*np.arange(len(hf))
        start=np.where(f_array == self.f_min)[0][0]
        end=np.where(f_array == self.f_max)[0][0]

        return hf[start: end + 1], f_array[start: end + 1]  


    def snr_squared_waveform(self, hp, hc, RA, Dec, psi, gmst, detector):
        """
        Calculates SNR squared of the simulated inspiral waveform for single detector 

        Parameters
        ----------
        hp, hc : lists
            plus and cross components of the frequency domain inspiral waveform
        RA, Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        detector : str
            name of detector in network (eg 'H1', 'L1')
        
        Returns
        -------
        float
            SNR squared
        
        """
        
        hf, f_array = self.simulate_waveform_response(hp, hc, RA, Dec, psi, gmst, detector)
        df = f_array[1]-f_array[0]
        SNR_squared=4*np.sum((np.abs(hf)**2/self.asds[detector](f_array)**2)*df)
        return SNR_squared

        
    def __reduced_amplitude(self, RA, Dec, inc, psi, detector, gmst):
        """
        Component of the Fourier amplitude, with redshift-dependent
        parts removed computes:
        [F+^2*(1+cos(i)^2)^2 + Fx^2*4*cos(i)^2]^1/2 * [5*pi/96]^1/2 * pi^-7/6

        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds

        Returns
        -------
        float
            Component of the Fourier amplitude, with
            redshift-dependent parts removed
        """
        Fplus = self.__Fplus(detector, RA, Dec, psi, gmst)
        Fcross = self.__Fcross(detector, RA, Dec, psi, gmst)
        return np.sqrt(Fplus**2*(1.0+np.cos(inc)**2)**2 + Fcross**2*4.0*np.cos(inc)**2)*np.sqrt(5.0*np.pi/96.0)*np.power(np.pi, -7.0/6.0)

    def __fmax(self, M):
        """
        Maximum frequency for integration, set by the frequency of
        the innermost stable orbit (ISO)
        fmax(M) 2*f_ISO = (6^(3/2)*pi*M)^-1

        Parameters
        ----------
        M : float
            total mass of the system in kg

        Returns
        -------
        float
            Maximum frequency in Hz
        """
        return 1/(np.power(6.0, 3.0/2.0)*np.pi*M) * lal.C_SI**3/lal.G_SI

    def __numfmax_fmax(self, M_min, detector):
        """
        lookup table for snr as a function of max frequency
        Calculates \int_fmin^fmax f'^(-7/3)/S_h(f')
        df over a range of values for fmax
        fmin: 10 Hz
        fmax(M): (6^(3/2)*pi*M)^-1
        and fmax varies from fmin to fmax(M_min)

        Parameters
        ----------
        M_min : float
            total minimum mass of the distribution in kg

        Returns
        -------
        Interpolated 1D array of \int_fmin^fmax f'^(-7/3)/S_h(f')
        for different fmax's
        """
        ASD = self.asds[detector]
        fmax = lambda m: self.__fmax(m)
        I = lambda f: np.power(f, -7.0/3.0)/(ASD(f)**2)
        f_min = self.f_min  # Hz, changed this from 20 to 10 to resolve NaN error
        f_max = fmax(M_min)

        arr_fmax = np.linspace(f_min, f_max, self.Nsamps)
        num_fmax = np.zeros(self.Nsamps)
        bar = progressbar.ProgressBar()
        print("Calculating lookup table for snr as a function of max frequency.")
        for i in bar(range(self.Nsamps)):
            num_fmax[i] = quad(I, f_min, arr_fmax[i], epsabs=0, epsrel=1.49e-4)[0]

        return interp1d(arr_fmax, num_fmax)
    

    def __snr_squared(self, RA, Dec, m1, m2, inc, psi, detector, gmst, z, H0):
        """
        the optimal snr squared for one detector, used for marginalising
        over sky location, inclination, polarisation, mass

        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        m1,m2 : float
            source masses in kg
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        z : float
            redshift
        H0 : float
            value of Hubble constant in kms-1Mpc-1

        Returns
        -------
        float
            snr squared for given parameters at a single detector
        """
        mtot = self.__mtot_obs(m1, m2, z)
        mc = self.mchirp_obs(m1, m2, z)
        A = self.__reduced_amplitude(RA, Dec, inc, psi, detector, gmst) * np.power(mc, 5.0/6.0) / (self.cosmo.dl_zH0(z, H0)*lal.PC_SI*1e6)

        fmax = self.__fmax(mtot)
        num = self.__interpolnum[detector](fmax)

        return 4.0*A**2*num*np.power(lal.G_SI, 5.0/3.0)/lal.C_SI**3.0

    def __pD_zH0(self, H0):
        """
        Detection probability over a range of redshifts and H0s,
        returned as an interpolated function.

        Parameters
        ----------
        H0 : float
            value of Hubble constant in kms-1Mpc-1

        Returns
        -------
        interpolated probabilities of detection over an array of
        luminosity distances, for a specific value of H0
        """
        lal_detectors = [lalsim.DetectorPrefixToLALDetector(name)
                                for name in self.detectors]
        
        network_rhosq = np.zeros((self.Nsamps, 1))
        prob = np.zeros(len(self.z_array))
        i=0
        bar = progressbar.ProgressBar()
        for z in bar(self.z_array):
            dl = self.cosmo.dl_zH0(z, H0)
            factor=1+z
            for n in range(self.Nsamps):
                if self.full_waveform is True: 
                    hp,hc = self.simulate_waveform(factor*self.m1[n], factor*self.m2[n], dl, self.incs[n], self.phis[n])
                    rhosqs = [self.snr_squared_waveform(hp,hc,self.RAs[n],self.Decs[n],self.psis[n], 0., det)
                              for det in self.detectors]

                else:
                    rhosqs = [self.__snr_squared(self.RAs[n], self.Decs[n],
                              self.m1[n], self.m2[n], self.incs[n], self.psis[n],
                              det, 0.0, self.z_array[i], H0)
                              for det in self.detectors]
                network_rhosq[n] = np.sum(rhosqs)

            survival = ncx2.sf(self.snr_threshold**2, 2*len(self.detectors), network_rhosq)  
            prob[i] = np.sum(survival, 0)/self.Nsamps
            i+=1
            
        return prob
    
    def __pD_zH0_array(self, H0vec):
        """
        Function which calculates p(D|z,H0) for a range of
        redshift and H0 values

        Parameters
        ----------
        H0vec : array_like
            array of H0 values in kms-1Mpc-1

        Returns
        -------
        list of arrays?
            redshift, H0 values, and the corresponding p(D|z,H0) for a grid
        """
        return np.array([self.__pD_zH0(H0) for H0 in H0vec])

    def pD_dlH0_eval(self, dl, H0):
        """
        Returns the probability of detection at a given value of
        luminosity distance and H0.
        Note that this is slower than the function pD_zH0_eval(z,H0).

        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc
        H0 : float
            H0 value in kms-1Mpc-1

        Returns
        -------
        float or array_like
            Probability of detection at the given luminosity distance and H0,
            marginalised over masses, inc, pol, and sky location
        """
        z = np.array([z_dlH0(x, H0) for x in dl])
        return expit(self.interp_average(z, H0))
    
    def pD_z_eval(self, z): 
        """
        Returns the probability of detection at a given value of
        redshift. To be used with Constant_H0 option set to True only. 

        Parameters
        ----------
        z : float or array_like
            value(s) of redshift
            
        Returns
        -------
        float or array_like
            Probability of detection at the given redshift,
            marginalised over masses, inc, pol, and sky location
        """
        return expit(self.interp_average(z))

    def pD_zH0_eval(self, z, H0):
        """
        Returns the probability of detection at a given value of
        redshift and H0.

        Parameters
        ----------
        z : float or array_like
            value(s) of redshift
        H0 : float
            H0 value in kms-1Mpc-1

        Returns
        -------
        float or array_like
            Probability of detection at the given redshift and H0,
            marginalised over masses, inc, pol, and sky location
        """
        return expit(self.interp_average(z,H0))
    
    def __call__(self, z, H0):
        """
        To call as function of z and H0

        Parameters
        ----------
        z : float or array_like
            value(s) of redshift
        H0 : float
            H0 value in kms-1Mpc-1

        Returns
        -------
        float or array_like
            Returns Pdet(z,H0).
        """
        return self.pD_zH0_eval(z, H0)

    def pD_distmax(self, dl, H0):
        """
        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc
        H0 : float
            H0 value in kms-1Mpc-1

        Returns
        -------
        float or array_like
            Returns twice the maximum distance given corresponding
            to Pdet(dl,H0) = 0.01.
        """
        return 2.*dl[np.where(self.pD_dlH0_eval(dl, H0) > 0.01)[0][-1]]

    def __snr_squared_basic(self, RA, Dec, m1, m2, inc, psi, detector, gmst, dl):
        """
        the optimal snr squared for one detector, used for marginalising over
        sky location, inclination, polarisation, mass
        Note that this ignores the redshifting of source masses.

        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        m1,m2 : float
            source masses in kg
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        dl : float
            luminosity distance in Mpc

        Returns
        -------
        float
            snr squared for given parameters at a single detector
        """
        mtot = self.__mtot(m1, m2)
        mc = self.mchirp(m1, m2)
        A = self.__reduced_amplitude(RA, Dec, inc, psi, detector, gmst) * np.power(mc, 5.0/6.0) / (dl*lal.PC_SI*1e6)

        fmax = self.__fmax(mtot)
        num = self.__interpolnum[detector](fmax)

        return 4.0*A**2*num*np.power(lal.G_SI, 5.0/3.0)/lal.C_SI**3.0

    def __pD_dl_basic(self):
        """
        Detection probability over a range of distances,
        returned as an interpolated function.
        Note that this ignores the redshifting of source masses.

        Returns
        -------
        interpolated probabilities of detection over an array of luminosity
        distances, for a specific value of H0.
        """
        
        rho = np.zeros((self.Nsamps, len(self.dl_array)))
        for n in range(self.Nsamps):
            rhosqs = [self.__snr_squared_basic(self.RAs[n], self.Decs[n], self.m1[n], self.m2[n], self.incs[n], self.psis[n], det, 0.0, self.dl_array) for det in self.detectors]
            rho[n] = np.sum(rhosqs, 0)

        survival = ncx2.sf(self.snr_threshold**2, 2*len(self.detectors), rho)
        prob = np.sum(survival, 0)/self.Nsamps
        self.spl = splrep(self.dl_array, prob)
        return splrep(self.dl_array, prob)

    def pD_dl_eval_basic(self, dl):
        """
        Returns a probability of detection at a given luminosity distance
        Note that this ignores the redshifting of source masses.

        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc

        Returns
        -------
        float or array_like
            Probability of detection at the given luminosity distance and H0,
            marginalised over masses, inc, pol, and sky location
        """
        return splev(dl, self.spl, ext=1)
