Detailed Neuron Models
=======================

Make Nengo use more realistic compartmental neurons simulated with [NEURON][1].

[1]: http://www.neuron.yale.edu/neuron/


Installation
============

Note that this project is still under development and the installation still
needs to be stream lined. Especially important is that there are several hard
coded paths at the moment which need to be adjusted.

The installation instructions in the following are for Linux systems and
probably work on OS X, too.

## Install InterViews

1. `wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/iv-19.tar.gz`
2. `tar xzf iv-19.tar.gz`
3. `cd iv-19`
4. `./configure --prefix=/usr/local`
5. `make`
6. `sudo make install`
7. `cd ..`

## Install NEURON

1. `wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.tar.gz`
2. `tar xzf nrn-7.4.tar.gz`
3. `cd nrn-7.4`
4. `./configure --prefix=/usr/local --with-iv=/usr/local --with-nrnpython`
   Add `--with-paranrn` if desired (needs openmpi).
5. `make`
6. `sudo make install`

### Install Python module

1. `cd src/nrnpython`
2. `python setup.py install`
3. `cd ../../..`

## Install nengo_detailed_neurons

1. `git clone https://github.com/nengo/nengo_detailed_neurons.git`
2. `cd nengo_detailed_neurons`
3. `python setup.py develop`
4. `cd models`
5. `/usr/local/x86_64/bin/nrnivmodl`
6. `cd ../BahlEtAl2012/channels`
7. `/usr/local/x86_64/bin/nrnivmodl`

Try the `communication_channel` notebook to see if it works.
