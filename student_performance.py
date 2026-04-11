"""
Student Academic Performance Analyzer
======================================
A linear algebra pipeline for analyzing, reducing, and predicting student 
academic performance using matrix decomposition techniques.

Uses: NumPy, SciPy, SymPy, Pandas
"""

import numpy as np
import pandas as pd
import scipy.linalg as la
import sympy as sp
import os


class StudentPerformanceAnalyzer:
    """
    Encapsulates a dataset of student scores and provides methods to perform
    various linear algebra techniques for analysis and prediction.
    """

    def __init__(self, data: np.ndarray, feature_names: list = None, student_ids: list = None):
        self.data = np.array(data, dtype=float)
        self.num_students, self.num_features = self.data.shape
        self.feature_names = feature_names or [f"Feature_{i+1}" for i in range(self.num_features)]
        self.student_ids = student_ids or [f"Student_{i+1}" for i in range(self.num_students)]

    # -- 1. RREF -----------------------------------------------------------
    def get_rref(self):
        """Compute RREF of the data matrix. Returns (rref_matrix, pivot_columns)."""
        sympy_matrix = sp.Matrix(self.data)
        rref_matrix, pivot_columns = sympy_matrix.rref()
        rref_numpy = np.array(rref_matrix.tolist(), dtype=float)
        return rref_numpy, list(pivot_columns)

    # -- 2. LU Decomposition -----------------------------------------------
    def get_lu_decomposition(self):
        """PA = LU decomposition. Returns (P, L, U)."""
        P, L, U = la.lu(self.data)
        return P, L, U

    def verify_lu(self, P, L, U):
        """Verify that P @ L @ U reconstructs the original data."""
        reconstructed = P @ L @ U
        error = np.linalg.norm(self.data - reconstructed)
        return error < 1e-8, error

    # -- 3. Rank & Nullity -------------------------------------------------
    def get_rank_and_nullity(self):
        """Returns (rank, nullity). Nullity = num_features - rank."""
        rank = np.linalg.matrix_rank(self.data)
        nullity = self.num_features - rank
        return rank, nullity

    # -- 4. Basis Selection ------------------------------------------------
    def get_basis(self):
        """Extract a basis for the column space using RREF pivot columns."""
        _, pivot_cols = self.get_rref()
        basis = self.data[:, pivot_cols]
        basis_features = [self.feature_names[i] for i in pivot_cols]
        redundant_features = [self.feature_names[i] for i in range(self.num_features) if i not in pivot_cols]
        return basis, basis_features, redundant_features

    # -- 5. Gram-Schmidt / QR -----------------------------------------------
    def get_orthogonal_basis(self):
        """Apply Gram-Schmidt (via QR) to the basis columns. Returns (Q, R)."""
        basis, _, _ = self.get_basis()
        Q, R = la.qr(basis, mode='economic')
        return Q, R

    def verify_orthogonality(self, Q):
        """Verify Q^T @ Q is approximately I."""
        product = Q.T @ Q
        identity = np.eye(Q.shape[1])
        error = np.linalg.norm(product - identity)
        return error < 1e-8, product, error

    # -- 6. Orthogonal Projection (Missing Score Prediction) ----------------
    def predict_missing_scores(self, student_vector, missing_indices):
        """
        Predict missing scores by learning a linear map from known to missing
        columns using least squares on the existing dataset.
        """
        student_vector = np.array(student_vector, dtype=float)
        known_indices = [i for i in range(self.num_features) if i not in missing_indices]

        A_known = self.data[:, known_indices]
        A_missing = self.data[:, missing_indices]

        X, _, _, _ = la.lstsq(A_known, A_missing)

        student_known = student_vector[known_indices]
        student_missing_pred = np.atleast_1d(student_known @ X)

        predicted_vector = student_vector.copy()
        for i, m_idx in enumerate(missing_indices):
            predicted_vector[m_idx] = student_missing_pred[i]

        return predicted_vector

    # -- 7. Least Squares Estimation ----------------------------------------
    def model_performance_trend(self, target_feature_idx):
        """
        Model one feature as a linear combination of the others via LSE.
        Returns (coefficients, predictor_indices, predictions, residuals, r_squared).
        """
        predictor_indices = [i for i in range(self.num_features) if i != target_feature_idx]
        X = self.data[:, predictor_indices]
        y = self.data[:, target_feature_idx]

        X_with_intercept = np.hstack([np.ones((self.num_students, 1)), X])
        coeffs, _, _, _ = la.lstsq(X_with_intercept, y)

        predictions = X_with_intercept @ coeffs
        residuals = y - predictions
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return coeffs, predictor_indices, predictions, residuals, r_squared

    # -- 8. Eigenvalue Decomposition / PCA ----------------------------------
    def discover_hidden_patterns(self):
        """
        Eigenvalue decomposition of the covariance matrix to discover 
        principal performance factors.
        """
        centered_data = self.data - np.mean(self.data, axis=0)
        cov_matrix = np.cov(centered_data, rowvar=False)

        eigenvalues, eigenvectors = la.eigh(cov_matrix)

        # Sort descending
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Variance explained ratio
        total_variance = np.sum(np.maximum(eigenvalues, 0))
        if total_variance > 0:
            variance_ratio = np.maximum(eigenvalues, 0) / total_variance
        else:
            variance_ratio = np.zeros_like(eigenvalues)

        # Project data onto top 2 principal components for 2D visualization
        top2 = eigenvectors[:, :2]
        projected_2d = centered_data @ top2

        return eigenvalues, eigenvectors, variance_ratio, projected_2d, cov_matrix

    # -- 9. Diagonalization -------------------------------------------------
    def diagonalize_covariance(self):
        """
        Diagonalize the covariance matrix: C = P D P^(-1).
        Returns (P, D, P_inv, is_diagonalizable).
        """
        centered = self.data - np.mean(self.data, axis=0)
        cov_matrix = np.cov(centered, rowvar=False)
        eigenvalues, P = la.eigh(cov_matrix)

        D = np.diag(eigenvalues)
        P_inv = np.linalg.inv(P)

        reconstructed = P @ D @ P_inv
        is_valid = np.allclose(cov_matrix, reconstructed, atol=1e-6)

        return P, D, P_inv, is_valid


def load_uci_student_data(filepath="data/student-mat.csv"):
    """
    Load the UCI Student Performance dataset and extract numeric features
    relevant for linear algebra analysis.
    
    Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
    Source: P. Cortez and A. Silva (2008). "Using Data Mining to Predict 
            Secondary School Student Performance."
    
    395 students, 33 attributes from two Portuguese schools.
    
    We select the following numeric features for our score matrix:
      - age: Student age
      - Medu: Mother's education (0-4)
      - Fedu: Father's education (0-4)
      - studytime: Weekly study time (1-4)
      - failures: Number of past class failures (0-3)
      - famrel: Quality of family relationships (1-5)
      - freetime: Free time after school (1-5)
      - goout: Going out with friends (1-5)
      - Dalc: Workday alcohol consumption (1-5)
      - Walc: Weekend alcohol consumption (1-5)
      - health: Current health status (1-5)
      - absences: Number of school absences
      - G1: First period grade (0-20)
      - G2: Second period grade (0-20)
      - G3: Final grade (0-20)
    
    Returns (data_array, feature_names, student_ids, df).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download it from: https://archive.ics.uci.edu/dataset/320/student+performance\n"
            "Place the CSV file at: data/student-mat.csv"
        )
    
    df = pd.read_csv(filepath)
    
    # Select numeric features meaningful for analysis
    numeric_features = [
        "age", "Medu", "Fedu", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc",
        "health", "absences", "G1", "G2", "G3"
    ]
    
    df_numeric = df[numeric_features]
    data_array = df_numeric.values.astype(float)
    student_ids = [f"S{i+1}" for i in range(len(df))]
    
    return data_array, numeric_features, student_ids, df
