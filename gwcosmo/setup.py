import sys
from setuptools import setup

setup_requires = ['setuptools >= 30.3.0']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    setup_requires.append('pytest-runner')
if {'build_sphinx'}.intersection(sys.argv):
    setup_requires.extend(['recommonmark',
                           'sphinx'])

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='gwcosmo',
      version='0.1.0',
      description='A package to estimate cosmological parameters using gravitational-wave observations',
      url='https://git.ligo.org/cbc-cosmo/gwcosmo',
      author='Cosmology R&D Group',
      author_email='cbc+cosmo@ligo.org',
      license='GNU',
      packages=['gwcosmo', 'gwcosmo.likelihood', 'gwcosmo.prior', 'gwcosmo.utilities'],
      package_dir={'gwcosmo': 'gwcosmo'},
      scripts=['bin/gwcosmo_single_posterior', 'bin/gwcosmo_combined_posterior', 'bin/gwcosmo_compute_pdet'],
      include_package_data=True,
      install_requires=[
          'numpy>=1.9',
          'matplotlib>=2.0',
          'pandas',
          'scipy',
          'tqdm'],
      setup_requires=setup_requires,
      zip_safe=False)

