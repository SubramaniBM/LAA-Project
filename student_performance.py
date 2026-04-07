import numpy as np
import scipy.linalg as la
import sympy as sp

class StudentPerformanceAnalyzer:
    def __init__(self, data: np.ndarray, subjects: list = None):
        """
        Initialize the analyzer with a dataset of student scores.
        data: 2D numpy array where rows are students and columns are subjects.
        subjects: List of subject names (optional).
        """
        self.data = np.array(data, dtype=float)
        self.num_students, self.num_subjects = self.data.shape
        self.subjects = subjects if subjects else [f"Subject_{i+1}" for i in range(self.num_subjects)]
    
    def get_rref(self):
        """
        Compute the Reduced Row Echelon Form (RREF) of the data matrix.
        Returns the RREF matrix and the indices of the pivot columns.
        """
        sympy_matrix = sp.Matrix(self.data)
        rref_matrix, pivot_columns = sympy_matrix.rref()
        # Convert sympy objects back to numpy float array
        rref_numpy = np.array(rref_matrix.tolist(), dtype=float)
        return rref_numpy, list(pivot_columns)
    
    def get_lu_decomposition(self):
        """
        Compute the LU decomposition of the dataset.
        Returns P, L, U matrices such that P @ L @ U = data.
        """
        P, L, U = la.lu(self.data)
        return P, L, U

    def get_rank_and_nullity(self):
        """
        Analyze the rank and nullity to understand the linear independence 
        of the subjects (columns).
        Returns rank, nullity.
        """
        rank = np.linalg.matrix_rank(self.data)
        nullity = self.num_subjects - rank
        return rank, nullity
        
    def get_basis(self):
        """
        Select a basis for the column space (non-redundant subjects) based on RREF pivot columns.
        Returns a matrix whose columns form a basis.
        """
        _, pivot_cols = self.get_rref()
        basis = self.data[:, pivot_cols]
        basis_subjects = [self.subjects[i] for i in pivot_cols]
        return basis, basis_subjects

    def get_orthogonal_basis(self):
        """
        Apply Gram-Schmidt orthogonalization to form an orthogonal subject basis
        from the pivot columns.
        Returns an orthogonal basis matrix Q.
        """
        basis, _ = self.get_basis()
        # Using QR decomposition which effectively implements Gram-Schmidt
        # Q contains the orthonormal column vectors
        Q, R = la.qr(basis, mode='economic')
        return Q

    def predict_missing_scores_projection(self, student_vector, missing_indices):
        """
        Predict missing scores using orthogonal projection onto the subspace 
        spanned by the known subjects.
        student_vector: 1D array of scores with np.nan for missing scores.
        missing_indices: List of indices where the score is missing.
        """
        student_vector = np.array(student_vector, dtype=float)
        known_indices = [i for i in range(self.num_subjects) if i not in missing_indices]
        
        # Submatrix of known features for all students (to form a basis for projection mapping)
        A_known = self.data[:, known_indices]
        # Target missing features for all students
        A_missing = self.data[:, missing_indices]
        
        # Find linear mapping (X) from known features to missing features using least squares
        # A_known @ X \approx A_missing
        X, residuals, rank, s = la.lstsq(A_known, A_missing)
        
        # Project current student's known scores to estimate the missing ones
        student_known = student_vector[known_indices]
        student_missing_pred = student_known @ X
        
        predicted_vector = student_vector.copy()
        for i, m_idx in enumerate(missing_indices):
            predicted_vector[m_idx] = student_missing_pred[i]
            
        return predicted_vector

    def model_performance_trend(self, target_subject_idx):
        """
        Use least squares estimation to model performance trends, e.g., predicting
        one specific subject as a linear combination of the other subjects based 
        on the dataset.
        Returns the coefficients of the model (intercept + weights).
        """
        predictor_indices = [i for i in range(self.num_subjects) if i != target_subject_idx]
        X = self.data[:, predictor_indices]
        y = self.data[:, target_subject_idx]
        
        # Add a column of ones for the intercept
        X_with_intercept = np.hstack([np.ones((self.num_students, 1)), X])
        
        coeffs, residuals, rank, s = la.lstsq(X_with_intercept, y)
        return coeffs, predictor_indices

    def discover_hidden_patterns(self):
        """
        Use Eigenvalue decomposition and diagonalization of the covariance matrix
        to discover hidden performance patterns (like PCA).
        Returns eigenvalues (variance explained) and eigenvectors (the patterns/factors).
        """
        # Center the data
        centered_data = self.data - np.mean(self.data, axis=0)
        
        # Compute covariance matrix (Subject x Subject)
        cov_matrix = np.cov(centered_data, rowvar=False)
        
        # Eigenvalue decomposition for symmetric matrices
        eigenvalues, eigenvectors = la.eigh(cov_matrix)
        
        # Sort in descending order of eigenvalue
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        return eigenvalues, eigenvectors
