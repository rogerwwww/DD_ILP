# DD_ILP

[![Build Status](https://travis-ci.org/pawelswoboda/DD_ILP.svg?branch=master)](https://travis-ci.org/pawelswoboda/DD_ILP)

DD_ILP is a framework for constructing integer linear problems pragmatically from dual decomposition based formulations. The resulting problems can be (approximately) solved with various backends, like ILP solvers or SAT-based ones. 
The interface provides various datatypes (single optimization variable, vectors, matrices and tensors of optimization variables) and convenience functions for creating popular constraints (logical constraints, simplex) for constructing optimization problems easily.
New backends can be added by providing wrappers for a few datatypes and constraint construction methods.
This project was originally developed for the [LP_MP](https://github.com/pawelswoboda/LP_MP.git) project.

## Backends
Current backends are
* Lingeling sat solver.
* export to text file (no optimization)

## Installation
Type `git clone https://github.com/pawelswoboda/DD_ILP.git` and `cd DD_ILP`. 
To initialize dependencies, type `git submodule update --init`.
Then configurate with `cmake .` and build with `make`.
