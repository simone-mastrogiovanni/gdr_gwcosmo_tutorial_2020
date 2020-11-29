# GWcosmo tutorial

This is a public repository containing the tutorial for using GWcosmo. This presentation is given at the [GdR meeting](https://indico.ijclab.in2p3.fr/event/6452/overview). Please follow the next steps to start the tutorial.

* Install and initialize [Anaconda](https://www.anaconda.com/products/individual) with a Python distribution >3.0. Anaconda allows you to manage in paralallel several Python distributions. This is very handy if you don't want to mess up with your main Python distribution. After that you installed Anaconda, you should open the terminal and check that Anaconda is initialized. If it is, you will read `(base)` close to your terminal location.
* Clone this repository with 
```
git clone https://github.com/simone-mastrogiovanni/gwcosmo_gdr_tutorial_2020.git
```
* The folder `gwcosmo` is a submodule linked to the current version of [Gwcosmo](https://git.ligo.org/lscsoft/gwcosmo). Now we need to install `gwcosmo`.
* After that you cloned this repository, we need to preapare a virtual environment for gwcosmo. Virtual environments are indipendent Python distributions, like a box. We do this by running 
```
 conda create -n gdrgwcosmo python=3.6
```
* Your virtual box is ready, now we access it by running 
```
conda activate gdrgwcosmo
```
if you want to go back to your main Python distribution, just close and open the terminal or run the same comand as before but with `base` instead of `gdrgwcosmo`
* Now that we are in `gdrgwcosmo` we need to install the packages needed to run with 
```
pip install -r requirements.txt
```
* Now we are ready to install gwcosmo by running
```
python gwcosmo/setup.py install
```
