"""
Visualizer Module
=================
Generates publication-quality charts for the linear algebra pipeline.
Uses Matplotlib and Seaborn with a dark theme.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

# Use a non-interactive backend so plots save without needing a display
matplotlib.use('Agg')

# -- Global Style ----------------------------------------------------------
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "font.family": "monospace",
})

OUTPUT_DIR = "plots"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. PCA Scatter Plot ---------------------------------------------------
def plot_pca_scatter(projected_2d, labels, variance_ratio, filename="pca_scatter.png"):
    """2D scatter of students projected onto top 2 principal components."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 7))

    unique_labels = list(dict.fromkeys(labels))
    colors = ["#58a6ff", "#f0883e", "#3fb950", "#f85149", "#bc8cff", "#e3b341"]
    markers = ["o", "s", "D", "^", "v", "p"]

    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        ax.scatter(
            projected_2d[mask, 0], projected_2d[mask, 1],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=80, alpha=0.75, edgecolors="white", linewidths=0.4,
            label=label, zorder=3
        )

    ax.set_xlabel(f"PC1 ({variance_ratio[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({variance_ratio[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("PCA -- Student Clusters in Eigenspace", fontweight="bold", fontsize=15)
    ax.legend(loc="best", framealpha=0.7, facecolor="#161b22", edgecolor="#30363d")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# -- 2. Correlation Heatmap ------------------------------------------------
def plot_correlation_heatmap(data, feature_names, filename="correlation_heatmap.png"):
    """Heatmap of the feature correlation matrix."""
    _ensure_output_dir()
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        xticklabels=feature_names, yticklabels=feature_names,
        cmap="coolwarm", center=0, linewidths=0.3,
        square=True, ax=ax,
        annot_kws={"fontsize": 7, "color": "white"},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=15, pad=15)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# -- 3. Eigenvalue Spectrum ------------------------------------------------
def plot_eigenvalue_spectrum(eigenvalues, variance_ratio, filename="eigenvalue_spectrum.png"):
    """Bar chart showing eigenvalues and cumulative explained variance."""
    _ensure_output_dir()
    n = len(eigenvalues)
    cumulative = np.cumsum(variance_ratio) * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(
        range(n), eigenvalues,
        color=["#58a6ff" if ev > 1 else "#8b949e" for ev in eigenvalues],
        edgecolor="#30363d", linewidth=0.8, alpha=0.85, zorder=3
    )
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Eigenvalue (Variance)", fontsize=12, color="#58a6ff")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([f"PC{i+1}" for i in range(n)], fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(range(n), cumulative, color="#f0883e", marker="o", linewidth=2, markersize=6, zorder=4)
    ax2.set_ylabel("Cumulative Variance %", fontsize=12, color="#f0883e")
    ax2.set_ylim(0, 110)
    ax2.axhline(y=95, color="#f85149", linestyle="--", alpha=0.5, label="95% threshold")
    ax2.legend(loc="center right", framealpha=0.7, facecolor="#161b22", edgecolor="#30363d")

    ax1.set_title("Eigenvalue Spectrum & Cumulative Variance", fontweight="bold", fontsize=15)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# -- 4. Least Squares Coefficients -----------------------------------------
def plot_lse_coefficients(coeffs, predictor_names, target_name, r_squared, filename="lse_coefficients.png"):
    """Bar chart of regression coefficients."""
    _ensure_output_dir()
    weights = coeffs[1:]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#3fb950" if w > 0 else "#f85149" for w in weights]
    bars = ax.barh(predictor_names, weights, color=bar_colors, edgecolor="#30363d", linewidth=0.8, alpha=0.85)

    ax.axvline(x=0, color="#8b949e", linewidth=1, linestyle="-")
    ax.set_xlabel("Coefficient Weight", fontsize=12)
    ax.set_title(
        f"Least Squares -- Predicting [{target_name}]  (R-sq = {r_squared:.3f})",
        fontweight="bold", fontsize=14
    )

    for bar, val in zip(bars, weights):
        x_pos = bar.get_width() + (0.005 if val >= 0 else -0.005)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                va="center", ha=ha, fontsize=9, color="#c9d1d9")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# -- 5. Missing Score Prediction Visual ------------------------------------
def plot_prediction_comparison(original, predicted, true_values, feature_names, filename="prediction_comparison.png"):
    """Side-by-side comparison of known vs predicted scores, including actuals for hidden."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(feature_names))
    width = 0.28

    known_mask = ~np.isnan(original)
    known_vals = np.where(known_mask, original, 0)
    
    # Truth values isolated to only the missing columns
    actual_hidden_vals = np.where(~known_mask, true_values, 0)

    ax.bar(x - width, known_vals, width, label="Known Scores",
           color="#58a6ff", edgecolor="#30363d", alpha=0.85)
           
    bars2 = ax.bar(x, predicted, width, label="Predicted Scores",
                   color="#3fb950", edgecolor="#30363d", alpha=0.85)
                   
    bars3 = ax.bar(x + width, actual_hidden_vals, width, label="Actual / Truth",
                   color="#bc8cff", edgecolor="#30363d", alpha=0.85)

    for i, is_known in enumerate(known_mask):
        if not is_known:
            # Highlight Predicted
            bars2[i].set_edgecolor("#f0883e")
            bars2[i].set_linewidth(2.5)
            ax.annotate("PREDICTED", (x[i], predicted[i] + 0.3),
                        ha="center", fontsize=7, color="#f0883e", fontweight="bold")
            # Highlight Actual
            bars3[i].set_edgecolor("#bc8cff")
            bars3[i].set_linewidth(2.5)
            ax.annotate("ACTUAL", (x[i] + width, actual_hidden_vals[i] + 0.3),
                        ha="center", fontsize=7, color="#bc8cff", fontweight="bold")

    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Missing Score Prediction via Orthogonal Projection", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=35, ha="right", fontsize=9)
    ax.legend(facecolor="#161b22", edgecolor="#30363d")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# -- 6. Grade Distribution ------------------------------------------------
def plot_grade_distributions(df, filename="grade_distributions.png"):
    """
    Histogram of G1, G2, G3 grade distributions from the original DataFrame.
    """
    _ensure_output_dir()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    grade_cols = ["G1", "G2", "G3"]
    colors = ["#58a6ff", "#f0883e", "#3fb950"]
    titles = ["First Period (G1)", "Second Period (G2)", "Final Grade (G3)"]

    for ax, col, color, title in zip(axes, grade_cols, colors, titles):
        ax.hist(df[col], bins=20, color=color, edgecolor="#30363d", alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xlabel("Grade (0-20)")
        ax.set_ylabel("Count")
        ax.axvline(df[col].mean(), color="#f85149", linestyle="--", linewidth=1.5,
                   label=f"Mean: {df[col].mean():.1f}")
        ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d")

    fig.suptitle("Grade Distributions -- UCI Student Performance Dataset",
                 fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# -- 7. Feature Importance from PCA ----------------------------------------
def plot_pca_feature_importance(eigenvectors, feature_names, filename="pca_feature_importance.png"):
    """
    Bar chart showing the contribution of each feature to the 
    top 3 principal components.
    """
    _ensure_output_dir()
    n_components = min(3, eigenvectors.shape[1])
    fig, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 6))
    if n_components == 1:
        axes = [axes]
    
    colors = ["#58a6ff", "#f0883e", "#3fb950"]
    
    for idx, ax in enumerate(axes):
        loadings = eigenvectors[:, idx]
        sorted_indices = np.argsort(np.abs(loadings))[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_loadings = loadings[sorted_indices]
        
        bar_colors = ["#3fb950" if l > 0 else "#f85149" for l in sorted_loadings]
        ax.barh(sorted_names[::-1], sorted_loadings[::-1], color=bar_colors[::-1],
                edgecolor="#30363d", alpha=0.85)
        ax.set_title(f"PC{idx+1} Feature Loadings", fontweight="bold", fontsize=13)
        ax.axvline(x=0, color="#8b949e", linewidth=1)
        ax.set_xlabel("Loading Weight")
    
    fig.suptitle("PCA Feature Importance (Eigenvector Loadings)",
                 fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
