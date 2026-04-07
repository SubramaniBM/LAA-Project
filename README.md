# Student Academic Performance Pipeline (LAA Project)

This project leverages fundamental Linear Algebra algorithms to analyze, simplify, predict, and discover latent patterns within student academic performance datasets. Built purely in Python, it serves as a demonstration of applying theoretical mathematical concepts to real-world matrices.

## Features & Linear Algebra Techniques Included

1. **Matrix Representation**: Encapsulates $M \times N$ datasets into dense NumPy array representations.
2. **Reduced Row Echelon Form (RREF)**: Evaluates row reductions to identify algebraically independent pivot traits (subjects).
3. **LU Decomposition**: Factors matrices into lower ($L$) and upper ($U$) triangular components ($P \cdot L \cdot U = A$).
4. **Rank and Nullity Analysis**: Determines redundant feature dimensions by algorithmically applying the Rank-Nullity Theorem ($Rank + Nullity = \text{Features}$).
5. **Basis Selection & Gram-Schmidt Orthogonalization (QR)**: Distills core independent traits and generates a fully orthonormal basis space ($Q^T \cdot Q = I$).
6. **Orthogonal Projection (Missing Score Estimation)**: Reverses independent variable mappings to algorithmically reconstruct missing student test scores using projection matrices.
7. **Least Squares Estimation (LSE)**: Finds optimal coefficient weights ($\hat{x}$) dynamically tracking performance trends of a specific subject by modeling out covariates.
8. **Eigenvalue Decomposition / Diagonalization**: Evaluates the Covariance matrix. Maps eigenvectors against their accompanying eigenvalues to discover hidden factors, principal performance vectors, and $0$ value variances.

## Project Structure

- `student_performance.py`: Contains the central object-oriented implementation class `StudentPerformanceAnalyzer`, utilizing arrays and linear mathematical processing blocks.
- `demo.py`: A self-contained simulation script holding a synthetic generated mathematical dataset embedding deliberate linear colinearities (Math + Physics = Science) for accurate mathematical demonstration.

## Setup and Installation

**Prerequisites:** 
You must have Python 3.x installed. 

This pipeline relies on mathematically targeted Python packages:
```bash
pip install numpy scipy sympy
```

## Running the Pipeline

To run the pipeline and see it in action, simply execute the demo script:
```bash
python demo.py
```

It will execute the dummy dataset directly through the analyzer class, actively feeding mathematical printouts to your terminal block-by-block. You can easily clone the code inside `demo.py` to pipe in your own real datasets via `pandas` or built-in arrays!
