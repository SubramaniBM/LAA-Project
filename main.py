"""
================================================================
  STUDENT ACADEMIC PERFORMANCE ANALYZER
  Linear Algebra Applied to Real Education Data
  Dataset: UCI Student Performance (P. Cortez, 2008)
================================================================
  An interactive terminal application that applies matrix
  decomposition, eigenanalysis, and projection to analyze,
  predict, and visualize student academic performance.
================================================================
"""

import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich import box
from rich.columns import Columns
from rich.text import Text

from student_performance import StudentPerformanceAnalyzer, load_uci_student_data
import visualizer as viz

console = Console(force_terminal=True)

# ── Global state ──────────────────────────────────────────────────────────
DATA = None
FEATURES = None
STUDENT_IDS = None
DF = None
ANALYZER = None

# Cache expensive computations
CACHE = {}


# ══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
def matrix_table(matrix, row_labels=None, col_labels=None, title="", precision=2):
    """Render a numpy matrix as a rich Table."""
    table = Table(title=title, box=box.SIMPLE_HEAVY, border_style="bright_cyan",
                  header_style="bold bright_cyan", show_lines=False, padding=(0, 1))
    if row_labels:
        table.add_column("", style="dim italic", no_wrap=True)
    cols = col_labels or [f"C{j}" for j in range(matrix.shape[1])]
    for c in cols:
        table.add_column(c, justify="right", style="white", no_wrap=True)
    rows_to_show = min(matrix.shape[0], 10)
    for i in range(rows_to_show):
        row_data = [f"{matrix[i, j]:.{precision}f}" for j in range(matrix.shape[1])]
        if row_labels:
            table.add_row(row_labels[i], *row_data)
        else:
            table.add_row(*row_data)
    if matrix.shape[0] > 10:
        filler = ["..."] * matrix.shape[1]
        if row_labels:
            table.add_row("...", *filler, style="dim")
        else:
            table.add_row(*filler, style="dim")
    return table


def section(title, step_num=None):
    console.print()
    if step_num is not None:
        console.rule(f"[bold bright_cyan]{step_num:02d}  {title}", style="bright_cyan")
    else:
        console.rule(f"[bold bright_cyan]{title}", style="bright_cyan")
    console.print()


def pause():
    Prompt.ask("\n  [dim]Press Enter to continue[/]", default="")


def show_feature_menu():
    feat_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    feat_table.add_column("#", justify="center", style="bright_cyan")
    feat_table.add_column("Feature", style="bold")
    feat_table.add_column("Description", style="dim")
    descriptions = {
        "age": "Student age (15-22)",
        "Medu": "Mother's education (0=none .. 4=higher)",
        "Fedu": "Father's education (0=none .. 4=higher)",
        "studytime": "Weekly study hours (1=<2h .. 4=>10h)",
        "failures": "Past class failures (0-3)",
        "famrel": "Family relationship quality (1-5)",
        "freetime": "Free time after school (1-5)",
        "goout": "Going out with friends (1-5)",
        "Dalc": "Workday alcohol consumption (1-5)",
        "Walc": "Weekend alcohol consumption (1-5)",
        "health": "Health status (1-5)",
        "absences": "Number of absences (0-75)",
        "G1": "First period grade (0-20)",
        "G2": "Second period grade (0-20)",
        "G3": "Final grade (0-20)",
    }
    for i, f in enumerate(FEATURES):
        feat_table.add_row(str(i), f, descriptions.get(f, ""))
    console.print(feat_table)


def load_data():
    """Load dataset into global state."""
    global DATA, FEATURES, STUDENT_IDS, DF, ANALYZER, CACHE
    DATA, FEATURES, STUDENT_IDS, DF = load_uci_student_data("data/student-mat.csv")
    ANALYZER = StudentPerformanceAnalyzer(DATA, FEATURES, STUDENT_IDS)
    CACHE = {}


# ══════════════════════════════════════════════════════════════════════════
#  MAIN MENU
# ══════════════════════════════════════════════════════════════════════════
def show_main_menu():
    console.print()
    menu = Table(box=box.ROUNDED, border_style="bright_cyan", show_header=False,
                 padding=(0, 2), title="[bold bright_white]MAIN MENU[/]")
    menu.add_column("Key", style="bold bright_cyan", justify="center", width=5)
    menu.add_column("Action", style="white")
    menu.add_column("", style="dim")

    menu.add_row("1", "Dataset Overview", "Browse the loaded dataset")
    menu.add_row("2", "Full Analysis Pipeline", "RREF, LU, Rank, Basis, QR (Steps 1-6)")
    menu.add_row("3", "Predict Missing Scores", "Hide features for a student and predict them")
    menu.add_row("4", "Predict YOUR Grade", "Enter your own data and get a prediction")
    menu.add_row("5", "Least Squares Modeling", "Model any feature as linear combination of others")
    menu.add_row("6", "PCA & Pattern Discovery", "Eigenvalue decomposition and visualization")
    menu.add_row("7", "Student Lookup & Compare", "Search and compare students side by side")
    menu.add_row("8", "Generate All Charts", "Save all 7 publication-quality plots")
    menu.add_row("", "", "")
    menu.add_row("0", "Exit", "")

    console.print(menu)


# ══════════════════════════════════════════════════════════════════════════
#  1. DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
def dataset_overview():
    section("DATASET OVERVIEW")

    summary = Table(title="UCI Student Performance -- Math Course",
                    box=box.ROUNDED, border_style="bright_yellow")
    summary.add_column("Statistic", style="bold")
    summary.add_column("Value", style="bright_cyan", justify="right")
    summary.add_row("Students", str(ANALYZER.num_students))
    summary.add_row("Features", str(ANALYZER.num_features))
    summary.add_row("Schools", "GP (Gabriel Pereira), MS (Mousinho da Silveira)")
    summary.add_row("Grade Scale", "0-20 (Portuguese system)")
    summary.add_row("Mean Final Grade (G3)", f"{DF['G3'].mean():.2f}")
    summary.add_row("Std Dev (G3)", f"{DF['G3'].std():.2f}")
    summary.add_row("Pass Rate (G3 >= 10)", f"{(DF['G3'] >= 10).mean()*100:.1f}%")
    summary.add_row("Highest G3", f"{DF['G3'].max()}")
    summary.add_row("Lowest G3", f"{DF['G3'].min()}")
    console.print(summary)

    console.print()
    show_feature_menu()

    # Ask how many students to view
    n = IntPrompt.ask("\n  How many students to display?", default=10)
    n = max(1, min(n, ANALYZER.num_students))
    console.print()
    console.print(matrix_table(DATA[:n, :], row_labels=STUDENT_IDS[:n],
                               col_labels=FEATURES,
                               title=f"Score Matrix A (first {n} students)", precision=1))

    pause()


# ══════════════════════════════════════════════════════════════════════════
#  2. FULL ANALYSIS PIPELINE (RREF, LU, Rank, Basis, QR)
# ══════════════════════════════════════════════════════════════════════════
def full_analysis_pipeline():
    # -- RREF --
    section("REDUCED ROW ECHELON FORM (RREF)", 1)
    console.print("  [dim]Computing RREF (this may take a moment)...[/]")

    if "rref" not in CACHE:
        CACHE["rref"] = ANALYZER.get_rref()
    rref_mat, pivot_cols = CACHE["rref"]

    pivot_names = [FEATURES[i] for i in pivot_cols]
    console.print(f"  Pivot columns: [bold bright_green]{pivot_cols}[/]")
    console.print(f"  Independent features: [bold bright_green]{', '.join(pivot_names)}[/]")
    redundant = [s for i, s in enumerate(FEATURES) if i not in pivot_cols]
    if redundant:
        console.print(f"  [bright_red]Redundant: {', '.join(redundant)}[/]")
    else:
        console.print(f"  [bright_green]No redundant features (all linearly independent)[/]")
    console.print()
    console.print(matrix_table(rref_mat[:8, :], col_labels=FEATURES,
                               title="RREF(A) [first 8 rows]", precision=2))
    pause()

    # -- LU --
    section("LU DECOMPOSITION", 2)
    P, L, U = ANALYZER.get_lu_decomposition()
    is_valid, err = ANALYZER.verify_lu(P, L, U)
    console.print(matrix_table(L[:6, :6], title="L (Lower Triangular) [6x6]", precision=3))
    console.print()
    console.print(matrix_table(U[:6, :], col_labels=FEATURES,
                               title="U (Upper Triangular) [6xN]", precision=2))
    status = "[bold bright_green][PASS][/]" if is_valid else "[bold bright_red][FAIL][/]"
    console.print(f"\n  P*L*U = A : {status}  (reconstruction error = {err:.2e})")
    pause()

    # -- Rank & Nullity --
    section("RANK & NULLITY", 3)
    rank, nullity = ANALYZER.get_rank_and_nullity()
    info = Table(box=box.ROUNDED, border_style="bright_yellow", show_header=False, padding=(0, 2))
    info.add_column("Property", style="bold")
    info.add_column("Value", style="bright_cyan", justify="right")
    info.add_row("Matrix Size", f"{ANALYZER.num_students} x {ANALYZER.num_features}")
    info.add_row("Rank", f"[bold bright_green]{rank}[/]")
    info.add_row("Nullity", f"[bold bright_red]{nullity}[/]")
    info.add_row("Rank + Nullity = Columns",
                 f"{rank} + {nullity} = {rank + nullity}  [dim]({ANALYZER.num_features} cols) [PASS][/]")
    console.print(info)
    pause()

    # -- Basis --
    section("BASIS SELECTION", 4)
    basis, basis_feats, redundant_feats = ANALYZER.get_basis()
    console.print(f"  [bold bright_green]Basis ({len(basis_feats)})[/]: {', '.join(basis_feats)}")
    if redundant_feats:
        console.print(f"  [bold bright_red]Redundant ({len(redundant_feats)})[/]: {', '.join(redundant_feats)}")
    console.print()
    console.print(matrix_table(basis[:8, :], row_labels=STUDENT_IDS[:8],
                               col_labels=basis_feats,
                               title="Basis Matrix (first 8 students)", precision=1))
    pause()

    # -- Gram-Schmidt / QR --
    section("GRAM-SCHMIDT ORTHOGONALIZATION (QR)", 5)
    Q, R = ANALYZER.get_orthogonal_basis()
    is_ortho, QtQ, ortho_err = ANALYZER.verify_orthogonality(Q)
    console.print(matrix_table(Q[:8, :min(8, Q.shape[1])],
                               title="Q Matrix (8x8 block)", precision=4))
    console.print()
    qtq_size = min(6, QtQ.shape[0])
    console.print(matrix_table(QtQ[:qtq_size, :qtq_size],
                               title=f"Q^T * Q [{qtq_size}x{qtq_size}] (should be I)",
                               precision=4))
    status = "[bold bright_green][PASS] ORTHOGONAL[/]" if is_ortho else "[bold bright_red][FAIL][/]"
    console.print(f"  {status}  (||Q'Q - I|| = {ortho_err:.2e})")
    pause()

    # Summary
    console.print(Panel(
        f"  [bright_cyan]*[/] Rank = [bold]{rank}[/], Nullity = [bold]{nullity}[/]\n"
        f"  [bright_cyan]*[/] LU reconstruction: [bold bright_green]PASS[/]\n"
        f"  [bright_cyan]*[/] Orthogonality: [bold bright_green]PASS[/]\n"
        f"  [bright_cyan]*[/] {len(basis_feats)} independent features identified",
        title="[bold]Analysis Summary[/]",
        border_style="bright_green", box=box.ROUNDED
    ))
    pause()


# ══════════════════════════════════════════════════════════════════════════
#  3. PREDICT MISSING SCORES
# ══════════════════════════════════════════════════════════════════════════
def predict_missing_scores():
    section("PREDICT MISSING SCORES")

    console.print("  [bold]Select a student from the dataset:[/]")
    console.print(f"  [dim]Enter a number between 1 and {ANALYZER.num_students}[/]")
    idx = IntPrompt.ask("  Student #", default=43) - 1
    idx = max(0, min(idx, ANALYZER.num_students - 1))

    student = DATA[idx].copy()
    console.print(f"\n  Selected: [bold bright_cyan]{STUDENT_IDS[idx]}[/]")

    # Show student's current scores
    stu_table = Table(box=box.ROUNDED, border_style="bright_yellow",
                      title=f"Scores for {STUDENT_IDS[idx]}")
    stu_table.add_column("Feature", style="bold")
    stu_table.add_column("Value", justify="center", style="bright_cyan")
    for i, f in enumerate(FEATURES):
        stu_table.add_row(f, f"{student[i]:.0f}")
    console.print(stu_table)

    # Choose features to hide
    console.print("\n  [bold]Which features should we hide and predict?[/]")
    show_feature_menu()
    missing_input = Prompt.ask(
        "  Feature numbers to predict (comma-separated)",
        default="13,14"
    )
    missing_idx = [int(x.strip()) for x in missing_input.split(",") if x.strip().isdigit()]
    missing_idx = [i for i in missing_idx if 0 <= i < ANALYZER.num_features]

    if not missing_idx:
        console.print("  [bright_red]Invalid selection, using G2 and G3[/]")
        missing_idx = [FEATURES.index("G2"), FEATURES.index("G3")]

    missing_names = [FEATURES[i] for i in missing_idx]
    console.print(f"\n  [bright_yellow]Hiding: {', '.join(missing_names)}[/]")

    actual_values = {i: student[i] for i in missing_idx}
    student_with_missing = student.copy()
    for i in missing_idx:
        student_with_missing[i] = np.nan

    predicted = ANALYZER.predict_missing_scores(student_with_missing, missing_idx)

    # Results table
    pred_table = Table(title=f"Prediction Results for {STUDENT_IDS[idx]}",
                       box=box.ROUNDED, border_style="bright_green")
    pred_table.add_column("Feature", style="bold")
    pred_table.add_column("Known", justify="center")
    pred_table.add_column("Predicted", justify="center")
    pred_table.add_column("Actual", justify="center")
    pred_table.add_column("Error", justify="center")
    for i, feat in enumerate(FEATURES):
        known = f"{student_with_missing[i]:.1f}" if not np.isnan(student_with_missing[i]) else "[dim]--[/]"
        pred = f"[bold bright_green]{predicted[i]:.1f}[/]" if i in missing_idx else f"{predicted[i]:.1f}"
        actual = f"[bold bright_yellow]{actual_values[i]:.1f}[/]" if i in missing_idx else ""
        err = f"{abs(predicted[i] - actual_values[i]):.2f}" if i in missing_idx else ""
        pred_table.add_row(feat, known, pred, actual, err)
    console.print(pred_table)

    avg_err = sum(abs(predicted[i] - actual_values[i]) for i in missing_idx) / len(missing_idx)
    console.print(f"  Average prediction error: [bold bright_cyan]{avg_err:.2f}[/] points (out of 20)")

    p = viz.plot_prediction_comparison(student_with_missing, predicted, FEATURES)
    console.print(f"  [dim]>> Chart saved -> [underline]{p}[/][/]")
    pause()


# ══════════════════════════════════════════════════════════════════════════
#  4. PREDICT YOUR GRADE  (the "wow" feature)
# ══════════════════════════════════════════════════════════════════════════
def predict_your_grade():
    section("PREDICT YOUR GRADE")

    console.print(Panel(
        "[bold bright_white]Enter your own data below.[/]\n"
        "The system will use the trained linear algebra model\n"
        "(Least Squares + Orthogonal Projection on 395 real students)\n"
        "to predict any feature you leave blank.\n\n"
        "[dim]Press Enter to skip a field (it will be predicted).[/]",
        border_style="bright_magenta", box=box.ROUNDED
    ))

    descriptions = {
        "age": ("Your age", "15-22", "17"),
        "Medu": ("Mother's education", "0=none, 1=primary, 2=middle, 3=secondary, 4=higher", ""),
        "Fedu": ("Father's education", "0=none, 1=primary, 2=middle, 3=secondary, 4=higher", ""),
        "studytime": ("Weekly study time", "1=<2h, 2=2-5h, 3=5-10h, 4=>10h", ""),
        "failures": ("Past class failures", "0-3", "0"),
        "famrel": ("Family relationship quality", "1=very bad .. 5=excellent", ""),
        "freetime": ("Free time after school", "1=very low .. 5=very high", ""),
        "goout": ("Going out with friends", "1=very low .. 5=very high", ""),
        "Dalc": ("Workday alcohol consumption", "1=very low .. 5=very high", ""),
        "Walc": ("Weekend alcohol consumption", "1=very low .. 5=very high", ""),
        "health": ("Current health status", "1=very bad .. 5=very good", ""),
        "absences": ("Number of school absences", "0-75", ""),
        "G1": ("First period grade", "0-20, leave blank to predict", ""),
        "G2": ("Second period grade", "0-20, leave blank to predict", ""),
        "G3": ("Final grade", "0-20, leave blank to predict", ""),
    }

    user_vector = np.full(len(FEATURES), np.nan)
    missing_idx = []

    console.print()
    for i, feat in enumerate(FEATURES):
        label, hint, default = descriptions[feat]
        prompt_text = f"  [{feat}] {label} [dim]({hint})[/]"
        val = Prompt.ask(prompt_text, default=default if default else "")
        val = val.strip()
        if val == "" or val.lower() in ("skip", "?", "predict"):
            missing_idx.append(i)
            console.print(f"    [dim]-> will be predicted[/]")
        else:
            try:
                user_vector[i] = float(val)
            except ValueError:
                missing_idx.append(i)
                console.print(f"    [dim]-> invalid input, will be predicted[/]")

    if not missing_idx:
        console.print("\n  [bright_yellow]You filled in everything! Nothing to predict.[/]")
        console.print("  [dim]Leave some fields blank next time to see predictions.[/]")
        pause()
        return

    if len(missing_idx) == len(FEATURES):
        console.print("\n  [bright_red]You skipped everything! Need at least some known values.[/]")
        pause()
        return

    known_names = [FEATURES[i] for i in range(len(FEATURES)) if i not in missing_idx]
    missing_names = [FEATURES[i] for i in missing_idx]

    console.print(f"\n  [bold]Known features ({len(known_names)})[/]: {', '.join(known_names)}")
    console.print(f"  [bold bright_magenta]Predicting ({len(missing_names)})[/]: {', '.join(missing_names)}")

    predicted = ANALYZER.predict_missing_scores(user_vector, missing_idx)

    # Results
    console.print()
    result_table = Table(title="YOUR PREDICTED PROFILE",
                         box=box.HEAVY, border_style="bright_magenta")
    result_table.add_column("Feature", style="bold")
    result_table.add_column("Your Input", justify="center")
    result_table.add_column("Predicted", justify="center")
    for i, feat in enumerate(FEATURES):
        if i in missing_idx:
            pred_val = predicted[i]
            # Color code grades
            if feat in ("G1", "G2", "G3"):
                if pred_val >= 16:
                    color = "bold bright_green"
                elif pred_val >= 10:
                    color = "bold bright_yellow"
                else:
                    color = "bold bright_red"
                result_table.add_row(feat, "[dim]--[/]", f"[{color}]{pred_val:.1f}[/]")
            else:
                result_table.add_row(feat, "[dim]--[/]", f"[bold bright_cyan]{pred_val:.1f}[/]")
        else:
            result_table.add_row(feat, f"{user_vector[i]:.0f}", f"{predicted[i]:.1f}")
    console.print(result_table)

    # If G3 was predicted, give a verdict
    g3_idx = FEATURES.index("G3")
    if g3_idx in missing_idx:
        g3_pred = predicted[g3_idx]
        if g3_pred >= 16:
            verdict = "[bold bright_green]EXCELLENT[/] -- Top performer!"
        elif g3_pred >= 14:
            verdict = "[bold bright_green]VERY GOOD[/] -- Well above average"
        elif g3_pred >= 12:
            verdict = "[bold bright_yellow]GOOD[/] -- Above average"
        elif g3_pred >= 10:
            verdict = "[bold bright_yellow]PASS[/] -- You should be fine"
        else:
            verdict = "[bold bright_red]AT RISK[/] -- Consider more study time"

        console.print(Panel(
            f"  Predicted Final Grade (G3): [bold]{g3_pred:.1f}[/] / 20\n"
            f"  Verdict: {verdict}\n\n"
            f"  [dim]Based on linear algebra model trained on {ANALYZER.num_students} real students[/]",
            title="[bold]Grade Prediction[/]",
            border_style="bright_magenta", box=box.DOUBLE_EDGE
        ))

    # Find most similar student in the dataset
    known_mask = ~np.isnan(user_vector)
    if np.any(known_mask):
        user_known = predicted[known_mask]
        data_known = DATA[:, known_mask]
        distances = np.linalg.norm(data_known - user_known, axis=1)
        closest_idx = np.argmin(distances)

        console.print(f"\n  [bold]Most similar student in dataset:[/] "
                      f"[bright_cyan]{STUDENT_IDS[closest_idx]}[/] "
                      f"(distance = {distances[closest_idx]:.2f})")

        sim_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1),
                          title=f"Comparison: You vs {STUDENT_IDS[closest_idx]}")
        sim_table.add_column("Feature", style="bold")
        sim_table.add_column("You", justify="center", style="bright_magenta")
        sim_table.add_column(STUDENT_IDS[closest_idx], justify="center", style="bright_cyan")
        for i, feat in enumerate(FEATURES):
            your_val = f"{predicted[i]:.1f}" if i in missing_idx else f"{user_vector[i]:.0f}"
            sim_table.add_row(feat, your_val, f"{DATA[closest_idx, i]:.0f}")
        console.print(sim_table)

    pause()


# ══════════════════════════════════════════════════════════════════════════
#  5. LEAST SQUARES MODELING
# ══════════════════════════════════════════════════════════════════════════
def least_squares_modeling():
    section("LEAST SQUARES ESTIMATION")

    console.print("  [bold]Choose a target feature to model:[/]")
    show_feature_menu()
    target_idx = IntPrompt.ask("  Target feature #", default=FEATURES.index("G3"))
    target_idx = max(0, min(target_idx, ANALYZER.num_features - 1))

    console.print(f"\n  Modeling [bold bright_magenta]{FEATURES[target_idx]}[/] "
                  f"= f(all other features)\n")

    coeffs, predictors, predictions, residuals, r_sq = ANALYZER.model_performance_trend(target_idx)

    predictor_names = [FEATURES[i] for i in predictors]
    lse_table = Table(title=f"Linear Model: {FEATURES[target_idx]}",
                      box=box.ROUNDED, border_style="bright_magenta")
    lse_table.add_column("Predictor", style="bold")
    lse_table.add_column("Coefficient", justify="right")
    lse_table.add_column("Impact", justify="center")

    lse_table.add_row("[dim]Intercept[/]", f"[bright_cyan]{coeffs[0]:.4f}[/]", "")
    for name, coef in zip(predictor_names, coeffs[1:]):
        color = "bright_green" if coef > 0 else "bright_red"
        # Impact bars
        bar_len = min(int(abs(coef) * 20), 20)
        bar = ("+" if coef > 0 else "-") * bar_len
        lse_table.add_row(name, f"[{color}]{coef:+.4f}[/]", f"[{color}]{bar}[/]")
    console.print(lse_table)

    console.print(f"\n  R-squared = [bold bright_cyan]{r_sq:.4f}[/]  "
                  f"([bold]{r_sq*100:.1f}%[/] of variance explained)")

    # Interpretation
    top_pos = [(predictor_names[i], coeffs[i+1]) for i in range(len(predictor_names)) if coeffs[i+1] > 0]
    top_neg = [(predictor_names[i], coeffs[i+1]) for i in range(len(predictor_names)) if coeffs[i+1] < 0]
    top_pos.sort(key=lambda x: x[1], reverse=True)
    top_neg.sort(key=lambda x: x[1])

    if top_pos:
        console.print(f"\n  [bright_green]Strongest positive factor:[/] "
                      f"[bold]{top_pos[0][0]}[/] ({top_pos[0][1]:+.4f})")
    if top_neg:
        console.print(f"  [bright_red]Strongest negative factor:[/] "
                      f"[bold]{top_neg[0][0]}[/] ({top_neg[0][1]:+.4f})")

    p = viz.plot_lse_coefficients(coeffs, predictor_names, FEATURES[target_idx], r_sq)
    console.print(f"\n  [dim]>> Chart saved -> [underline]{p}[/][/]")
    pause()


# ══════════════════════════════════════════════════════════════════════════
#  6. PCA & PATTERN DISCOVERY
# ══════════════════════════════════════════════════════════════════════════
def pca_analysis():
    section("PCA & PATTERN DISCOVERY")

    eigenvalues, eigenvectors, var_ratio, proj_2d, cov_mat = ANALYZER.discover_hidden_patterns()

    eig_table = Table(title="Eigenvalue Spectrum", box=box.ROUNDED, border_style="bright_cyan")
    eig_table.add_column("PC", justify="center", style="bold")
    eig_table.add_column("Eigenvalue", justify="right")
    eig_table.add_column("Variance %", justify="right")
    eig_table.add_column("Cumulative %", justify="right")
    eig_table.add_column("", style="bright_cyan")

    cumulative = 0
    for i in range(len(eigenvalues)):
        cumulative += var_ratio[i] * 100
        eig_color = "bright_green" if var_ratio[i] > 0.05 else "dim"
        bar_len = int(var_ratio[i] * 40)
        bar = "#" * bar_len
        eig_table.add_row(
            f"PC{i+1}",
            f"[{eig_color}]{eigenvalues[i]:.2f}[/]",
            f"{var_ratio[i]*100:.1f}%",
            f"{cumulative:.1f}%",
            f"[{eig_color}]{bar}[/]"
        )
    console.print(eig_table)

    # Top eigenvector interpretation
    console.print(f"\n  [bold]PC1 Loadings[/] (what drives the most variance):")
    loading_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    loading_table.add_column("Feature", style="bold")
    loading_table.add_column("Loading", justify="right")
    loading_table.add_column("", style="bright_cyan")

    sorted_idx = np.argsort(np.abs(eigenvectors[:, 0]))[::-1]
    for i in sorted_idx:
        val = eigenvectors[i, 0]
        color = "bright_green" if val > 0 else "bright_red"
        bar_len = int(abs(val) * 30)
        bar = ("#" if val > 0 else "-") * bar_len
        loading_table.add_row(FEATURES[i], f"[{color}]{val:+.4f}[/]", f"[{color}]{bar}[/]")
    console.print(loading_table)

    # Diagonalization
    P_mat, D_mat, P_inv, diag_valid = ANALYZER.diagonalize_covariance()
    diag_status = "[bold bright_green][PASS][/]" if diag_valid else "[bold bright_red][FAIL][/]"
    console.print(f"\n  Diagonalization C = P*D*P^(-1): {diag_status}")

    # Find how many PCs needed for 95%
    cum = np.cumsum(var_ratio) * 100
    n_95 = np.argmax(cum >= 95) + 1
    console.print(f"  Components for 95% variance: [bold bright_cyan]{n_95}[/] out of {len(eigenvalues)}")
    console.print(f"  Dimensionality reduction: [bold]{len(eigenvalues)}D -> {n_95}D[/] "
                  f"([bright_green]{(1 - n_95/len(eigenvalues))*100:.0f}% reduction[/])")

    pause()


# ══════════════════════════════════════════════════════════════════════════
#  7. STUDENT LOOKUP & COMPARE
# ══════════════════════════════════════════════════════════════════════════
def student_lookup():
    section("STUDENT LOOKUP & COMPARE")

    console.print("  [bold]Enter two student numbers to compare side-by-side:[/]")
    console.print(f"  [dim]Range: 1 to {ANALYZER.num_students}[/]\n")

    idx_a = IntPrompt.ask("  Student A #", default=1) - 1
    idx_b = IntPrompt.ask("  Student B #", default=100) - 1
    idx_a = max(0, min(idx_a, ANALYZER.num_students - 1))
    idx_b = max(0, min(idx_b, ANALYZER.num_students - 1))

    student_a = DATA[idx_a]
    student_b = DATA[idx_b]

    # Side-by-side comparison
    cmp_table = Table(title=f"{STUDENT_IDS[idx_a]} vs {STUDENT_IDS[idx_b]}",
                      box=box.ROUNDED, border_style="bright_cyan")
    cmp_table.add_column("Feature", style="bold")
    cmp_table.add_column(STUDENT_IDS[idx_a], justify="center", style="bright_cyan")
    cmp_table.add_column(STUDENT_IDS[idx_b], justify="center", style="bright_magenta")
    cmp_table.add_column("Diff", justify="center")

    for i, feat in enumerate(FEATURES):
        diff = student_a[i] - student_b[i]
        if abs(diff) < 0.01:
            diff_str = "[dim]==[/]"
        elif diff > 0:
            diff_str = f"[bright_green]+{diff:.0f}[/]"
        else:
            diff_str = f"[bright_red]{diff:.0f}[/]"
        cmp_table.add_row(feat, f"{student_a[i]:.0f}", f"{student_b[i]:.0f}", diff_str)
    console.print(cmp_table)

    # Similarity metrics
    # Euclidean distance
    euclidean = np.linalg.norm(student_a - student_b)

    # Cosine similarity
    cos_sim = np.dot(student_a, student_b) / (np.linalg.norm(student_a) * np.linalg.norm(student_b))

    # PCA projection comparison
    eigenvalues, eigenvectors, var_ratio, proj_2d, _ = ANALYZER.discover_hidden_patterns()
    centered = DATA - np.mean(DATA, axis=0)
    proj_a = centered[idx_a] @ eigenvectors[:, :2]
    proj_b = centered[idx_b] @ eigenvectors[:, :2]
    eigenspace_dist = np.linalg.norm(proj_a - proj_b)

    sim_table = Table(title="Similarity Metrics", box=box.ROUNDED, border_style="bright_yellow")
    sim_table.add_column("Metric", style="bold")
    sim_table.add_column("Value", justify="right", style="bright_cyan")
    sim_table.add_column("Interpretation", style="dim")
    sim_table.add_row("Euclidean Distance", f"{euclidean:.2f}",
                      "Lower = more similar")
    sim_table.add_row("Cosine Similarity", f"{cos_sim:.4f}",
                      "1.0 = identical direction")
    sim_table.add_row("Eigenspace Distance (2D)", f"{eigenspace_dist:.2f}",
                      "Distance in PC1-PC2 plane")
    console.print(sim_table)

    # Position in PCA space
    console.print(f"\n  [dim]{STUDENT_IDS[idx_a]} in PCA space: ({proj_a[0]:.1f}, {proj_a[1]:.1f})[/]")
    console.print(f"  [dim]{STUDENT_IDS[idx_b]} in PCA space: ({proj_b[0]:.1f}, {proj_b[1]:.1f})[/]")

    pause()


# ══════════════════════════════════════════════════════════════════════════
#  8. GENERATE ALL CHARTS
# ══════════════════════════════════════════════════════════════════════════
def generate_all_charts():
    section("GENERATING CHARTS")

    console.print("  [dim]Saving 7 publication-quality charts to plots/ ...[/]\n")

    # Grade distributions
    console.print("  [1/7] Grade distributions...", end=" ")
    p = viz.plot_grade_distributions(DF)
    console.print(f"[bright_green]OK[/] -> {p}")

    # Correlation heatmap
    console.print("  [2/7] Correlation heatmap...", end=" ")
    p = viz.plot_correlation_heatmap(DATA, FEATURES)
    console.print(f"[bright_green]OK[/] -> {p}")

    # PCA
    eigenvalues, eigenvectors, var_ratio, proj_2d, _ = ANALYZER.discover_hidden_patterns()
    grade_labels = []
    for g in DF["G3"]:
        if g >= 16:
            grade_labels.append("Excellent (16-20)")
        elif g >= 12:
            grade_labels.append("Good (12-15)")
        elif g >= 10:
            grade_labels.append("Pass (10-11)")
        else:
            grade_labels.append("Fail (0-9)")

    console.print("  [3/7] PCA scatter...", end=" ")
    p = viz.plot_pca_scatter(proj_2d, grade_labels, var_ratio)
    console.print(f"[bright_green]OK[/] -> {p}")

    console.print("  [4/7] Eigenvalue spectrum...", end=" ")
    p = viz.plot_eigenvalue_spectrum(eigenvalues, var_ratio)
    console.print(f"[bright_green]OK[/] -> {p}")

    console.print("  [5/7] PCA feature importance...", end=" ")
    p = viz.plot_pca_feature_importance(eigenvectors, FEATURES)
    console.print(f"[bright_green]OK[/] -> {p}")

    # LSE
    target_idx = FEATURES.index("G3")
    coeffs, predictors, _, _, r_sq = ANALYZER.model_performance_trend(target_idx)
    predictor_names = [FEATURES[i] for i in predictors]

    console.print("  [6/7] LSE coefficients...", end=" ")
    p = viz.plot_lse_coefficients(coeffs, predictor_names, "G3", r_sq)
    console.print(f"[bright_green]OK[/] -> {p}")

    # Prediction comparison
    test_student = DATA[42].copy()
    missing_idx = [FEATURES.index("G2"), FEATURES.index("G3")]
    student_missing = test_student.copy()
    for i in missing_idx:
        student_missing[i] = np.nan
    predicted = ANALYZER.predict_missing_scores(student_missing, missing_idx)

    console.print("  [7/7] Prediction comparison...", end=" ")
    p = viz.plot_prediction_comparison(student_missing, predicted, FEATURES)
    console.print(f"[bright_green]OK[/] -> {p}")

    console.print(Panel(
        "[bold bright_green]All 7 charts saved to plots/[/]",
        border_style="bright_green", box=box.ROUNDED
    ))
    pause()


# ══════════════════════════════════════════════════════════════════════════
#  APPLICATION LOOP
# ══════════════════════════════════════════════════════════════════════════
def main():
    # Header
    console.print(Panel.fit(
        "[bold bright_white]STUDENT ACADEMIC PERFORMANCE ANALYZER[/]\n"
        "[dim]Linear Algebra Applied to Real Education Data[/]\n\n"
        "[bright_cyan]Dataset[/]: UCI Student Performance (Cortez & Silva, 2008)\n"
        "[bright_cyan]Source[/] : archive.ics.uci.edu/dataset/320\n"
        "[bright_cyan]Students[/]: 395 | [bright_cyan]Features[/]: 15 | "
        "[bright_cyan]Techniques[/]: 10",
        border_style="bright_cyan", box=box.DOUBLE_EDGE,
        padding=(1, 4)
    ))

    # Load data
    console.print("\n  [dim]Loading dataset...[/]", end=" ")
    load_data()
    console.print(f"[bright_green]OK[/] ({ANALYZER.num_students} students loaded)\n")

    # Main loop
    while True:
        show_main_menu()

        choice = Prompt.ask("\n  [bold bright_cyan]>[/] Choose an option", default="0")
        choice = choice.strip()

        if choice == "1":
            dataset_overview()
        elif choice == "2":
            full_analysis_pipeline()
        elif choice == "3":
            predict_missing_scores()
        elif choice == "4":
            predict_your_grade()
        elif choice == "5":
            least_squares_modeling()
        elif choice == "6":
            pca_analysis()
        elif choice == "7":
            student_lookup()
        elif choice == "8":
            generate_all_charts()
        elif choice == "0":
            console.print("\n  [dim]Goodbye![/]\n")
            break
        else:
            console.print("  [bright_red]Invalid option. Try 0-8.[/]")


if __name__ == "__main__":
    main()
