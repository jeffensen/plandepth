# Adaptive planning depth
Investigating adaptive planning depth using a multi stage task


Requirements
------------

numpy
pandas
matplotlib
seaborn
pytorch
pyro
numpyro

Installation
------------

The easiest way to install required libraries is using [conda](https://conda.io/miniconda.html)
and pip package managers.

First setup an environment using anaconda prompt (or just terminal in linux):

```sh
conda create -n ppl python=3 numpy pandas matplotlib seaborn
conda activate ppl
conda install pytorch -c pytorch
pip install pyro-ppl
```

Next we will install numpyro
```sh
conda install fastcache -c conda-forge
pip install numpyro
```