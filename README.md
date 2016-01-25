Tensor Factorization via Matrix Factorization
=============================================

Tensor factorization is a key subroutine in several
recent algorithms for learning latent variable models using the method of moments. 
This general technique is applicable to a broad class of models,
such as:

* Mixtures of Gaussians
* Topic models (e.g. latent Dirichlet allocation)
* Hidden Markov models

However, techniques for factorizing tensors are not as well-developed as 
matrix factorization techniques. The algorithms implemented here instead 
transform the problem of finding the CP decomposition of a
tensor to the problem of jointly diagonalizing a set of matrices.

These ideas have been proposed and analyzed in the following publications:

```
V. Kuleshov, A. Chaganty, and P. Liang. Tensor Factorization via Matrix Factorization. AISTATS 2015
V. Kuleshov, A. Chaganty, and P. Liang. Simultaneous Diagonalization: the asymmetric, low-rank, and noisy settings. ArXiv Technical report.
```

## Installation

### Requirements

The algorithms have been implemented in MATLAB and make extensive use of:

* MATLAB Tensor Toolbox 2.5
* Tensorlab 2.02

These libraries are available for free for academic use.

Unfortunately, it seems like the current version of Octave (3.8.2)
does not support the Tensor Toolbox, which means our code cannot be 
used in Octave.

### Setup

To install this package, simply clone the git repo:

```
git clone [...];
cd tenfact;
```

You must then make sure that Tensorlab and the Tensor Toolbox can be seen from 
MATLAB (i.e. make sure to run `addpath` on their paths).

## Contents

The main algorithms are in `/bin`. The exact scripts are:

* `jacobi.m`: Jacobi algorithm for simultaneous matrix diagonalization.
* `qrj1d.m`: QRJ1D algorithm for the non-orthogonal case.
* `tenfact.m`: Orthogonal tensor factorization using the `OJD0/OJD1` algorithms from the paper.
* `no_tenfact.m`: Non-orthogonal tensor factorization using the `NOJD0/NOJD1` algorithms.
* `tpm.m`: Our implementation of the tensor power method.

The root folder contains files for reproducing the synthetic experiments 
from the paper.

## Reproducing experiments from the paper

To reproduce the orthogonal experiments, use the script `run_ortho_comparison.m`. This 
sets certain global parameters (e.g., the dimension, the rank, the noise level, etc.)
and for each level of noise, it performs a series of synthetic 
experiments (implemented in `run_ortho_experiment.m`). 

Each synthetic experiment is defined by:

* The tensor dimension `p`
* The tensor rank `k`
* The noise level `epsilon`
* The number of experiment repetitions `tries`
* An output file `outputfile`

The results reported in `outputfile` are averged over all the repetitions.

The scripts `run_nonortho_comparison.m` and `run_nonortho_experiment.m` perform the 
same analysis for non-orthogonal tensors.

The result of one experiment (in `outputfile`) has the following format:
```
nojd0   0.142636        128.800000
nojd1   0.135456        244.100000
als     0.200180        404.800000
lath    0.235587        12.000000
nls     0.154352        167.800000
```

The first column is the name of the algorithm used, which is one of:

* `ojd0/ojd1`: our orthogonal tensor factorization algorithms
* `nojd0/nojd1`: our non-orthogonal tensor factorization algorithms
* `tpm`: the tensor power method
* `als`: alternating least squares
* `nls`: non-linear least squares
* `lath`: Lathauwers algorithm

The implementations of most of these algorithms are taken from Tensorlab.

The second column in the output file is the error of the algorithm.
The last column measures the running time. For the joint diagonalization 
algorithms, it measures the total number of sweeps by the JD subroutine. 
For the TPM, it measures the total number of multiplications. See the 
documentation in Tensorlab for the other algorithms.

## Feedback

Please send feedback to [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov)
