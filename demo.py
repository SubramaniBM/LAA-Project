import numpy as np
from student_performance import StudentPerformanceAnalyzer

def main():
    print("=== Student Academic Performance Pipeline (Linear Algebra Demo) ===")
    
    # 1. Create a synthetic dataset
    # Subjects: Math, Physics, Chemistry, Literature, History, Science (Redundant = Math + Physics)
    subjects = ["Math", "Physics", "Chemistry", "Literature", "History", "Science_Dependent"]
    
    # Generate some base data for 10 students
    np.random.seed(42)
    # Math, Physics, Chemistry, Literature, History
    base_data = np.random.randint(50, 100, size=(10, 5))
    
    # Science is exactly Math + Physics (creates linear dependency for RREF / rank demo)
    science_scores = base_data[:, 0] + base_data[:, 1]
    
    # Combine into a single matrix
    data = np.column_stack([base_data, science_scores])
    
    analyzer = StudentPerformanceAnalyzer(data, subjects)
    
    print("\n--- Initial Data Matrix (10 Students x 6 Subjects) ---")
    print(analyzer.data)
    
    # 2. RREF & Pivot Columns
    rref_mat, pivot_cols = analyzer.get_rref()
    print("\n--- Reduced Row Echelon Form (RREF) ---")
    print(np.round(rref_mat, 2))
    print(f"Pivot Columns (Independent Subjects): {pivot_cols}")
    
    # 3. LU Decomposition
    P, L, U = analyzer.get_lu_decomposition()
    print("\n--- LU Decomposition ---")
    print("Matrix L (Lower Triangular Base):")
    print(np.round(L, 2))
    print("Matrix U (Upper Triangular Factor):")
    print(np.round(U, 2))
    
    # 4. Rank and Nullity
    rank, nullity = analyzer.get_rank_and_nullity()
    print(f"\n--- Rank and Nullity ---")
    print(f"Rank (Number of independent features): {rank}")
    print(f"Nullity (Number of redundant features): {nullity}")
    
    # 5. Basis Selection
    basis, basis_subjects = analyzer.get_basis()
    print("\n--- Basis for Column Space ---")
    print(f"Selected Subjects: {basis_subjects}")
    print(basis)
    
    # 6. Gram-Schmidt Orthogonalization
    Q = analyzer.get_orthogonal_basis()
    print("\n--- Orthogonal Basis (Gram-Schmidt) Q ---")
    print(np.round(Q, 4))
    
    # Verify orthogonality (Q^T * Q should be close to Identity)
    print("Verify Q^T @ Q (Should be Identity Matrix):")
    print(np.round(Q.T @ Q, 2))
    
    # 7. Orthogonal Projection for missing scores
    # Let's say a student has missing Physics (idx 1) and Chemistry (idx 2) scores
    # Let's assume their actual scores would have been roughly Physics=80, Chemistry=90
    # Thus Science = 85 + 80 = 165
    student_scores = np.array([85, np.nan, np.nan, 78, 82, 165]) 
    missing_indices = [1, 2]
    
    print("\n--- Orthogonal Projection (Predicting Missing Scores) ---")
    print(f"New student incoming scores: {student_scores}")
    predicted_scores = analyzer.predict_missing_scores_projection(student_scores, missing_indices)
    print(f"Predicted full scores: {np.round(predicted_scores, 2)}")
    
    # 8. Least Squares Estimation (Performance Trend)
    # Target: Chemistry (idx 2). Can we model it using other subjects?
    coeffs, predictors = analyzer.model_performance_trend(target_subject_idx=2)
    print("\n--- Least Squares Estimation ---")
    print("Modeling Chemistry score based on other subjects.")
    print(f"Intercept: {coeffs[0]:.2f}")
    for idx, coef in zip(predictors, coeffs[1:]):
        print(f"Coefficient for {subjects[idx]}: {coef:.2f}")
        
    # 9. Eigenvalue Decomposition / Diagonalization (Patterns)
    eigenvalues, eigenvectors = analyzer.discover_hidden_patterns()
    print("\n--- Eigenvalue Decomposition (PCA-like Hidden Patterns) ---")
    print(f"Eigenvalues (Explained Variance): {np.round(eigenvalues, 2)}")
    print("Top Principal Eigenvector (Primary Performance Factor):")
    print(np.round(eigenvectors[:, 0], 2))

if __name__ == "__main__":
    main()
