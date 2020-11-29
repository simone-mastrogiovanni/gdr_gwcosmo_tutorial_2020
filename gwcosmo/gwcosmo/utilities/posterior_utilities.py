# Global Imports
import numpy as np
import gwcosmo
from scipy.integrate import cumtrapz
from scipy.optimize import fmin
from scipy.interpolate import splev, splrep, interp1d
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import leastsq


class confidence_interval(object):
    def __init__(self, posterior, H0, level=0.683, verbose=False):
        self.posterior = posterior
        self.H0 = H0
        self.level = level
        self.verbose = verbose
        self.lower_level, self.upper_level = self.HDI()
        self.interval = self.upper_level - self.lower_level
        self.map = self.MAP()
        
    def HDI(self):
        cdfvals = cumtrapz(self.posterior, self.H0)
        sel = cdfvals > 0.
        x = self.H0[1:][sel]
        cdfvals = cdfvals[sel]
        ppf = interp1d(cdfvals, x, fill_value=0., bounds_error=False)

        def intervalWidth(lowTailPr):
            ret = ppf(self.level + lowTailPr) - ppf(lowTailPr)
            if (ret > 0.):
                return ret
            else:
                return 1e4
        HDI_lowTailPr = fmin(intervalWidth, 1. - self.level, disp=self.verbose)[0]
        return ppf(HDI_lowTailPr), ppf(HDI_lowTailPr + self.level)


    def MAP(self):
        sp = UnivariateSpline(self.H0, self.posterior, s=0.)
        x_highres = np.linspace(self.H0[0], self.H0[-1], 100000)
        y_highres = sp(x_highres)
        return x_highres[np.argmax(y_highres)]
