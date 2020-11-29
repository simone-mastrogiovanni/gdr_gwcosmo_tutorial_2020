"""
Rachel Gray 2020
"""

class SchechterParams():
    """
    Returns the source frame Schechter function parameters for a given band.
    ugriz parameters are from https://arxiv.org/pdf/astro-ph/0210215.pdf
        (tables 1 and 2)
    
    Parameters
    ----------
    band : observation band (B,K,u,g,r,i,z)
    """
    def __init__(self, band):
        self.Mstar = None
        self.alpha = None
        self.Mmin = None
        self.Mmax = None
        
        self.alpha, self.Mstar, self.Mmin, self.Mmax = self.values(band)
        
    def values(self, band):
        if band == 'B':
            return -1.07, -19.70, -22.96, -12.96
        elif band == 'K':
            return -1.02, -22.77, -27.0, -12.96
        elif band == 'u':
            return -0.92, -17.93, -21.93, -15.54 #TODO check Mmin and Mmax
        elif band == 'g':
            return -0.89, -19.39, -23.38, -16.10 #TODO check Mmin and Mmax
        elif band == 'r':
            return -1.05, -20.44, -24.26, -16.11 #TODO check Mmin and Mmax
        elif band == 'i':
            return -1.00, -20.82, -23.84, -17.07 #TODO check Mmin and Mmax
        elif band == 'z':
            return -1.08, -21.18, -24.08, -17.34 #TODO check Mmin and Mmax
        else:
            raise Exception("Expected 'B', 'K', 'u', 'g', 'r', 'i' or 'z' band argument") 
        

