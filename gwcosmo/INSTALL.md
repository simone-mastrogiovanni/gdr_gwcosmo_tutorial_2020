# Quick start

## To install

GWCosmo requires Python >= 3.5.

The easiest way to install it is with `virtualenv` and `pip`:

	$ virtualenv --system-site-packages ~/gwcosmo
	$ source ~/gwcosmo/bin/activate
	$ pip install git+https://git.ligo.org/cbc-cosmo/gwcosmo

*  **Note:** GWCosmo requires a fairly new version of `setuptools`. If you get
   an error message that looks like this:

       pkg_resources.VersionConflict: (setuptools 0.9.8 (gwcelery/lib/python2.7/site-packages), Requirement.parse('setuptools>=30.3.0'))

   then run `pip install --upgrade setuptools` and try again.


## To test

With `setup.py`:

	$ python setup.py test

