# Code Explanation Document

> A detailed breakdown of every module, class, function, and the linear algebra theory behind the Student Academic Performance Analyzer.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [student_performance.py -- Core Analyzer](#student_performancepy----core-analyzer)
3. [visualizer.py -- Chart Generator](#visualizerpy----chart-generator)
4. [main.py -- Interactive Application](#mainpy----interactive-application)
5. [Mathematical Theory](#mathematical-theory)

---

## Project Overview

The project applies **10 linear algebra techniques** to a real-world dataset of 395 students to:
- Analyze relationships between academic features
- Predict missing grades using matrix projection
- Discover hidden performance patterns via eigenanalysis
- Model final grades as linear combinations of other factors

**Data Flow:**
```
CSV File (395 x 33)
    |
    v
load_uci_student_data() --> Extracts 15 numeric features --> NumPy matrix (395 x 15)
    |
    v
StudentPerformanceAnalyzer --> Applies all 10 linear algebra techniques
    |
    v
main.py (interactive menu) + visualizer.py (7 charts)
```

---

## student_performance.py -- Core Analyzer

This file contains the `StudentPerformanceAnalyzer` class and the data loading function. Every method implements a specific linear algebra operation.

### `load_uci_student_data(filepath)`

**Purpose:** Loads the UCI Student Performance CSV and extracts 15 numeric columns.

**How it works:**
1. Reads the CSV using `pandas.read_csv()`
2. Selects only the numeric columns relevant for analysis:
   - Demographic: `age`, `Medu` (mother's education), `Fedu` (father's education)
   - Behavioral: `studytime`, `failures`, `freetime`, `goout`, `Dalc`, `Walc`
   - Other: `famrel`, `health`, `absences`
   - Grades: `G1` (period 1), `G2` (period 2), `G3` (final)
3. Converts the DataFrame to a NumPy float array
4. Returns the matrix, feature names, student IDs, and the full DataFrame

```python
df = pd.read_csv(filepath)
numeric_features = ["age", "Medu", "Fedu", ..., "G1", "G2", "G3"]
data_array = df[numeric_features].values.astype(float)
```

---

### `class StudentPerformanceAnalyzer`

Initialized with a data matrix `A` of shape `(m x n)` where:
- `m` = number of students (rows) = 395
- `n` = number of features (columns) = 15

---

### `get_rref()` -- Reduced Row Echelon Form

**What it does:** Transforms matrix A into its simplest row-equivalent form using elementary row operations.

**Why we use it:** RREF reveals which columns (features) are **pivot columns** (linearly independent) and which are **free columns** (redundant -- expressible as combinations of others).

**Implementation:**
```python
sympy_matrix = sp.Matrix(self.data)       # Convert to SymPy for exact arithmetic
rref_matrix, pivot_columns = sympy_matrix.rref()  # Gauss-Jordan elimination
```

**Key insight:** We use SymPy (not NumPy) because RREF requires exact rational arithmetic. Floating-point errors in NumPy would produce incorrect pivot detection.

**Output:** For this real dataset, all 15 columns are pivot columns (Rank = 15), meaning no feature is a perfect linear combination of others.

---

### `get_lu_decomposition()` -- LU Factorization

**What it does:** Factors the matrix as `PA = LU` where:
- `P` = permutation matrix (row swaps)
- `L` = lower triangular (multipliers from elimination)
- `U` = upper triangular (result of elimination)

**Why we use it:** LU decomposition is the computational backbone of solving systems `Ax = b`. It's more efficient than RREF for repeated solves with different right-hand sides.

**Implementation:**
```python
P, L, U = scipy.linalg.lu(self.data)
```

**Verification:** We check that `P @ L @ U` reconstructs the original matrix:
```python
error = np.linalg.norm(self.data - P @ L @ U)  # Should be ~0
```

---

### `get_rank_and_nullity()` -- Rank-Nullity Theorem

**What it does:** Computes:
- **Rank** = number of linearly independent columns = dimension of column space
- **Nullity** = dimension of null space = number of "free" features

**The Rank-Nullity Theorem states:** `rank(A) + nullity(A) = n` (number of columns)

**Implementation:**
```python
rank = np.linalg.matrix_rank(self.data)  # Uses SVD internally
nullity = self.num_features - rank
```

**Why this matters:** If nullity > 0, some features are redundant and can be removed without losing information. For our dataset: rank = 15, nullity = 0, meaning all features carry unique information.

---

### `get_basis()` -- Basis for the Column Space

**What it does:** Extracts the columns corresponding to RREF pivot positions. These columns form a **basis** -- a minimal set of independent vectors that span the column space.

**Implementation:**
```python
_, pivot_cols = self.get_rref()
basis = self.data[:, pivot_cols]  # Select only pivot columns
```

**Why we use it:** A basis is the most compact representation of the data's column space. Redundant features (non-pivot columns) can be expressed as linear combinations of basis vectors.

---

### `get_orthogonal_basis()` -- Gram-Schmidt Process via QR

**What it does:** Takes the basis columns and produces an **orthonormal basis** Q where:
- Every column has unit length (||q_i|| = 1)
- Every pair of columns is perpendicular (q_i · q_j = 0 for i ≠ j)

**Implementation:**
```python
Q, R = scipy.linalg.qr(basis, mode='economic')
# Q = orthonormal columns, R = upper triangular coefficients
```

**How QR relates to Gram-Schmidt:**
The QR factorization is mathematically equivalent to the Gram-Schmidt process but is numerically more stable. The Gram-Schmidt process takes vectors v_1, v_2, ..., v_n and produces orthonormal vectors:
```
u_1 = v_1 / ||v_1||
u_2 = (v_2 - proj(v_2 onto u_1)) / ||...||
u_3 = (v_3 - proj(v_3 onto u_1) - proj(v_3 onto u_2)) / ||...||
```

**Verification:** We check Q^T × Q = I (the identity matrix):
```python
product = Q.T @ Q
error = np.linalg.norm(product - np.eye(Q.shape[1]))  # Should be ~0
```

---

### `predict_missing_scores(student_vector, missing_indices)` -- Orthogonal Projection

**What it does:** Given a student with some known features and some missing features, it predicts the missing values using the relationship learned from all 395 students.

**The Math:**
Let A_known be the columns of A corresponding to known features, and A_missing be the columns for missing features. We find a linear map X such that:

```
A_known × X ≈ A_missing   (least squares fit)
```

Then for a new student with known values `s_known`:
```
predicted_missing = s_known × X
```

**Implementation:**
```python
X, _, _, _ = scipy.linalg.lstsq(A_known, A_missing)  # Solve for mapping
student_missing_pred = student_known @ X               # Apply to new student
```

**Why this is "orthogonal projection":** The least squares solution minimizes the distance between A_known × X and A_missing. Geometrically, this is an orthogonal projection of A_missing onto the column space of A_known.

---

### `model_performance_trend(target_feature_idx)` -- Least Squares Estimation

**What it does:** Models one feature (e.g., Final Grade G3) as a linear combination of all other features:

```
G3 ≈ β₀ + β₁(age) + β₂(Medu) + β₃(Fedu) + ... + β₁₄(G2)
```

**The Math:**
We solve the overdetermined system `Xβ = y` where:
- X = feature matrix with an added column of 1s (for intercept)
- y = target feature column
- β = coefficient vector (what we solve for)

The **Normal Equation** solution is: `β = (X^T X)^(-1) X^T y`

In practice we use `lstsq` which is numerically more stable.

**R-squared calculation:**
```python
R² = 1 - (SS_residual / SS_total)
   = 1 - (Σ(y - ŷ)² / Σ(y - ȳ)²)
```
R² = 0.837 for G3 means the model explains 83.7% of grade variance.

**Key finding:** G2 has the largest coefficient (+0.97), meaning the second period grade is by far the best predictor of the final grade.

---

### `discover_hidden_patterns()` -- Eigenvalue Decomposition / PCA

**What it does:** Discovers the principal directions of variation in the data by decomposing the **covariance matrix**.

**Step-by-step:**
1. **Center the data:** Subtract the mean of each column
   ```python
   centered = data - np.mean(data, axis=0)
   ```
2. **Compute covariance matrix:** C = (1/n) × centered^T × centered
   ```python
   cov_matrix = np.cov(centered, rowvar=False)  # 15x15 symmetric matrix
   ```
3. **Eigendecomposition:** Find eigenvalues λ and eigenvectors v such that `Cv = λv`
   ```python
   eigenvalues, eigenvectors = scipy.linalg.eigh(cov_matrix)
   ```
4. **Sort by descending eigenvalue** (largest variance first)
5. **Project data onto top 2 eigenvectors** for 2D visualization

**Why `eigh` not `eig`?** The covariance matrix is symmetric and positive semi-definite, so `eigh` (Hermitian eigendecomposition) is both faster and guarantees real eigenvalues.

**Variance ratio:** Each eigenvalue represents the variance captured by that principal component:
```python
variance_ratio[i] = eigenvalue[i] / sum(all eigenvalues)
```

**Result:** PC1 explains 52.3% of variance (driven by `absences`), PC2 explains 34.4% (driven by grades). Together they capture 86.6%.

---

### `diagonalize_covariance()` -- Matrix Diagonalization

**What it does:** Verifies the diagonalization `C = P D P⁻¹` where:
- C = covariance matrix (15x15)
- P = matrix of eigenvectors (columns)
- D = diagonal matrix of eigenvalues
- P⁻¹ = inverse of P

**Why this works:** A symmetric matrix is always diagonalizable, and its eigenvectors form an orthogonal matrix (P⁻¹ = P^T).

**Verification:**
```python
reconstructed = P @ D @ P_inv
is_valid = np.allclose(cov_matrix, reconstructed, atol=1e-6)
```

---

## visualizer.py -- Chart Generator

Uses Matplotlib and Seaborn with a custom dark theme. All charts are saved as PNG files to the `plots/` folder.

### Global Dark Theme
```python
plt.rcParams.update({
    "figure.facecolor": "#0e1117",   # Dark background
    "axes.facecolor": "#161b22",     # Slightly lighter plot area
    "text.color": "#c9d1d9",         # Light gray text
    ...
})
```

### Chart Functions

| Function | What It Plots | Key Details |
|----------|---------------|-------------|
| `plot_pca_scatter` | Students in 2D eigenspace | Color-coded by grade tier (Fail/Pass/Good/Excellent) |
| `plot_correlation_heatmap` | 15x15 correlation matrix | Uses `np.corrcoef()`, annotated with values |
| `plot_eigenvalue_spectrum` | Bar chart + cumulative line | Dual y-axis, 95% threshold line |
| `plot_lse_coefficients` | Horizontal bar chart | Green=positive, Red=negative coefficients |
| `plot_prediction_comparison` | Known vs predicted side-by-side | Orange border highlights predicted features |
| `plot_grade_distributions` | G1, G2, G3 histograms | Mean line overlay, 3 subplots |
| `plot_pca_feature_importance` | Eigenvector loadings for PC1-3 | Sorted by absolute loading magnitude |

---

## main.py -- Interactive Application

### Architecture
The application follows a **menu loop** pattern:
```
main() -> load_data() -> while True: show_menu() -> handle_choice()
```

### Global State
Data is loaded once into global variables to avoid reloading:
```python
DATA = None       # 395x15 NumPy array
FEATURES = None   # List of 15 feature names
ANALYZER = None   # StudentPerformanceAnalyzer instance
CACHE = {}        # Stores expensive results (like RREF)
```

### The "Predict YOUR Grade" Algorithm
1. User enters values for some features, leaves others blank
2. Known values go into a vector, blanks become `NaN`
3. `predict_missing_scores()` uses least squares to predict the blanks
4. The most similar real student is found via minimum Euclidean distance:
   ```python
   distances = np.linalg.norm(DATA - user_vector, axis=1)
   closest = np.argmin(distances)
   ```

### Student Comparison Metrics
| Metric | Formula | Meaning |
|--------|---------|---------|
| Euclidean Distance | `||a - b||₂ = √(Σ(aᵢ - bᵢ)²)` | Raw difference in feature space |
| Cosine Similarity | `(a · b) / (||a|| × ||b||)` | Direction similarity (1.0 = identical direction) |
| Eigenspace Distance | `||Pa - Pb||₂` where P = top-2 PCs | Distance in the PCA-reduced 2D space |

---

## Mathematical Theory

### Why Linear Algebra for Student Data?

Each student is a **vector** in 15-dimensional space. The entire class is a **matrix** where:
- Rows = students (observations)
- Columns = features (variables)

Linear algebra lets us:
1. **Simplify** the matrix (RREF, LU) to understand its structure
2. **Measure** information content (Rank, Nullity)
3. **Orthogonalize** features (QR) to remove correlations
4. **Project** into subspaces to predict missing data
5. **Decompose** variance (Eigenvalues) to find what matters most

### Connection Between Techniques

```
Raw Data Matrix A (395 x 15)
    |
    |-- RREF --> Pivot columns --> Basis Selection
    |                                    |
    |-- LU Decomposition                 v
    |                            Gram-Schmidt (QR)
    |-- Rank/Nullity                     |
    |                                    v
    |                        Orthogonal Projection
    |                         (Missing Score Prediction)
    |
    |-- Covariance Matrix C = (1/n) A^T A
            |
            |-- Eigendecomposition --> Eigenvalues (variance)
            |                      --> Eigenvectors (directions)
            |
            |-- Diagonalization: C = PDP^(-1)
            |
            |-- PCA: Project onto top eigenvectors
            |
            |-- Least Squares: Ax ≈ b (minimize ||Ax - b||²)
```

### Key Results from This Dataset

| Finding | Technique Used | Value |
|---------|---------------|-------|
| All 15 features are independent | Rank Analysis | rank = 15, nullity = 0 |
| G2 is the strongest predictor of G3 | Least Squares | coefficient = +0.97 |
| `absences` drives the most variance | PCA (PC1) | 52.3% of total variance |
| 7 components capture 95% of information | Eigenvalue Analysis | 15D reduced to 7D |
| Missing grades predicted within ~1 point | Orthogonal Projection | avg error ≈ 0.78 |
| LSE model explains 83.7% of grade variance | R-squared | R² = 0.837 |
