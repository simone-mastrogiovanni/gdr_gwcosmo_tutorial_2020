# Contributing

Contributors may familiarize themselves with GWCosmo itself by going through the
[First Steps with GWCosmo](https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Cosmology/gwcosmo/outline) tutorial.

# Working with large files

The best and recommended way to work with large files and git is to use Git lfs

To get started with Git lfs:

Download and install the Git lfs command line extension. You only have to set up Git LFS once using Homebrew or Macports:

	$ brew install git-lfs
	$ port install git-lfs

Then cd into the gwcosmo directory and do:

	$ git lfs install

Select the file types that you want Git LFS to manage (these go into the .gitattributes file):

	$ git lfs track "*.gwf"

Make sure .gitattributes is tracked:

	$ git add .gitattributes

There is no step three. Just commit and push to GitHub as you normally would.

	$ git add ./data/catalog_data/DES.dat
	$ git commit -m "Added DES catalogue file"
	$ git push origin master