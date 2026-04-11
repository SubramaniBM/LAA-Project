# Student Academic Performance Analyzer

> **Linear Algebra Applied to Real Education Data** -- An interactive terminal application that uses matrix decomposition, eigenanalysis, and projection to analyze, predict, and visualize student academic performance.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-blue?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.x-orange?logo=scipy)

---

## Dataset

**UCI Student Performance Dataset** (P. Cortez and A. Silva, 2008)

- **Source**: [archive.ics.uci.edu/dataset/320](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Description**: 395 students from two Portuguese secondary schools
- **33 attributes** including demographics, social factors, and academic grades
- **Grade scale**: 0-20 (Portuguese grading system)
- **Grade periods**: G1 (first), G2 (second), G3 (final)

We extract **15 numeric features**: `age`, `Medu`, `Fedu`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`, `G1`, `G2`, `G3`.

---

## Features

### Interactive Menu
A persistent menu-driven interface lets you explore any analysis on demand:

| Option | Feature | Description |
|--------|---------|-------------|
| 1 | **Dataset Overview** | Browse students, view statistics, adjustable display count |
| 2 | **Full Analysis Pipeline** | RREF, LU Decomposition, Rank/Nullity, Basis Selection, Gram-Schmidt QR |
| 3 | **Predict Missing Scores** | Choose any student, hide any features, compare predictions to actuals |
| 4 | **Predict YOUR Grade** | Enter your own demographics and grades to get a live prediction |
| 5 | **Least Squares Modeling** | Model any feature as a linear combination of others with R-squared |
| 6 | **PCA & Pattern Discovery** | Eigenvalue decomposition, variance analysis, dimensionality reduction |
| 7 | **Student Lookup & Compare** | Side-by-side comparison with cosine similarity and eigenspace distance |
| 8 | **Generate All Charts** | Save 7 publication-quality dark-themed plots to `plots/` |

### Linear Algebra Techniques

| Technique | Application |
|-----------|-------------|
| **RREF** | Identify linearly independent features |
| **LU Decomposition** | Factor score matrix with reconstruction verification |
| **Rank-Nullity Theorem** | Quantify feature redundancy |
| **Basis Selection** | Extract minimal independent feature set |
| **Gram-Schmidt / QR** | Orthonormal basis with orthogonality proof |
| **Orthogonal Projection** | Predict missing grades from known features |
| **Least Squares Estimation** | Linear regression with R-squared reporting |
| **Eigenvalue Decomposition** | PCA for hidden pattern discovery |
| **Diagonalization** | Verify C = PDP^(-1) on covariance matrix |
| **Cosine Similarity** | Measure student similarity in feature space |

### Visualizations (7 charts)

- Grade Distributions (G1, G2, G3 histograms)
- PCA Scatter (students by grade tier in eigenspace)
- Correlation Heatmap (15x15 feature correlation)
- Eigenvalue Spectrum (variance explained per PC)
- PCA Feature Importance (eigenvector loadings)
- LSE Coefficients (regression weights for G3)
- Prediction Comparison (known vs predicted scores)

## Quick Start

```bash
pip install numpy scipy sympy rich matplotlib seaborn pandas
python main.py
```

## Project Structure

```
LAA Project/
|-- main.py                  # Interactive terminal application
|-- student_performance.py   # Core analyzer class + UCI data loader
|-- visualizer.py            # Matplotlib/Seaborn chart generators
|-- data/
|   |-- student-mat.csv      # UCI Student Performance dataset
|-- plots/                   # Generated charts (auto-created)
|-- README.md
```

## Citation

P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* In A. Brito and J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008.
