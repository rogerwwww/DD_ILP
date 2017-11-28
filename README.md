# DD_ILP
DD_ILP is a framework for constructing integer linear problems pragmatically from dual decomposition based formulations. The resulting problems can be (approximately) solved with various backends, like ILP solvers or SAT-based ones. 
The interface provides various datatypes (single optimization variable, vectors, matrices and tensors of optimization variables) and convenience functions for creating popular constraints (logical constraints, simplex) for constructing optimization problems easily.
New backends can be added by providing wrappers for a few datatypes and constraint construction methods.
This project was originally developed for the [LP_MP](https://github.com/pawelswoboda/LP_MP.git) project.

## Backends
Current backends are
* Lingeling sat solver.
* Gurobi.
* export to text file (no optimization)
