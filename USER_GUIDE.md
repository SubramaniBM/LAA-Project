# User Guide

A step-by-step guide to using the Student Academic Performance Analyzer.

---

## Getting Started

### 1. Install Dependencies (one time only)
```bash
pip install numpy scipy sympy rich matplotlib seaborn pandas
```

### 2. Run the Application
```bash
python main.py
```

You'll see a welcome banner and then the **Main Menu**.

---

## The Main Menu

After loading, you'll see 8 numbered options. Type a number and press Enter.

```
┌─────┬────────────────────────┬────────────────────────┐
│  1  │  Dataset Overview      │  See the data          │
│  2  │  Full Analysis         │  Run all math steps    │
│  3  │  Predict Missing       │  Predict for a student │
│  4  │  Predict YOUR Grade    │  Enter your own data   │
│  5  │  Least Squares         │  Model a feature       │
│  6  │  PCA & Patterns        │  Find hidden patterns  │
│  7  │  Compare Students      │  Side-by-side compare  │
│  8  │  Generate Charts       │  Save 7 plot images    │
│  0  │  Exit                  │                        │
└─────┴────────────────────────┴────────────────────────┘
```

You can run options in **any order**, as many times as you want. After each option finishes, you return to this menu.

---

## Option-by-Option Guide

### Option 1: Dataset Overview

**What it shows:** Summary statistics of the 395 students and the raw data table.

**What it asks:**
```
How many students to display? (10):
```
Type a number (e.g., `20`) or just press Enter for the default (10).

**Good for:** Getting familiar with the data before doing any analysis.

---

### Option 2: Full Analysis Pipeline

**What it shows:** All the core linear algebra steps, one after another:
1. **RREF** — Which features are independent
2. **LU Decomposition** — L and U matrices with reconstruction check
3. **Rank & Nullity** — How many features carry unique information
4. **Basis Selection** — The minimal set of independent features
5. **Gram-Schmidt (QR)** — Orthonormal basis with Q^T × Q = I proof

**What it asks:** Nothing — just press Enter to advance through each step.

**What to look for:**
- `[PASS] VERIFIED` badges (green = the math checks out)
- Rank = 15, Nullity = 0 (all features are independent)
- Q^T × Q matrix should look like an identity matrix (1s on diagonal, 0s elsewhere)

---

### Option 3: Predict Missing Scores

**What it does:** Picks a student from the dataset, hides some of their scores, predicts them, and compares to the actual values.

**What it asks (step by step):**

1. **"Student #"** — Pick a student number between 1 and 395.
   ```
   Student # (43): 150
   ```
   The app shows that student's actual scores.

2. **"Feature numbers to predict"** — A numbered list of features appears. Enter the numbers of the ones you want to hide, separated by commas.
   ```
   Enter feature numbers to predict (comma-separated) (13,14): 12,13,14
   ```
   This would hide G1, G2, and G3 (grades) and predict them.

**What to look for:**
- The **Predicted** column vs the **Actual** column
- The **Error** column (lower = better prediction)
- Average error is typically under 1-2 points on the 0-20 scale

**Tip:** The defaults (student 43, predicting G2 and G3) give a great result. Just press Enter twice.

---

### Option 4: Predict YOUR Grade ⭐

**What it does:** You enter your own information and the system predicts any fields you leave blank.

**What it asks:** 15 questions, one for each feature. Each shows the valid range:
```
[age] Your age (15-22) (17): 18
[Medu] Mother's education (0=none .. 4=higher) (): 3
[studytime] Weekly study time (1=<2h .. 4=>10h) (): 2
...
[G1] First period grade (0-20, leave blank to predict) (): 12
[G2] Second period grade (0-20, leave blank to predict) (): 14
[G3] Final grade (0-20, leave blank to predict) ():     <- leave blank!
```

**Rules:**
- Type a number and press Enter → that value is kept
- Just press Enter (blank) → that value will be **predicted**
- You must fill in **at least some** fields and leave **at least one** blank

**What you get:**
- A table with your input + predicted values
- A **verdict** if G3 was predicted (Excellent / Very Good / Good / Pass / At Risk)
- The **most similar real student** in the dataset with a side-by-side comparison

**Tip for demos:** Leave only G3 blank and fill everything else. The prediction is surprisingly accurate.

---

### Option 5: Least Squares Modeling

**What it does:** Models one feature as a weighted sum of all other features.

**What it asks:**
```
Target feature # (14): 
```
A numbered list appears. Enter the number of the feature you want to model. Default is 14 (G3 = Final Grade).

**What to look for:**
- **Coefficient table** — positive (green) means that feature increases the target; negative (red) means it decreases it
- **Impact bars** — `++++++` or `------` show strength visually
- **R-squared** — how much variance the model explains (0.84 = 84% for G3)
- **Strongest factor** — G2 is the strongest predictor of G3

**Fun experiment:** Try modeling `absences` (feature 11) as the target — you'll see a much lower R² because absences are harder to predict from grades.

---

### Option 6: PCA & Pattern Discovery

**What it does:** Eigenvalue decomposition to find which "directions" in the data carry the most information.

**What it asks:** Nothing — just shows results.

**What to look for:**
- **Eigenvalue table** — PC1 has the largest eigenvalue (most variance)
- **Visual bars** — `########` show relative size of each component
- **PC1 Loadings** — which features contribute most to the primary pattern
- **95% threshold** — how many components you need to capture 95% of information
- **Dimensionality reduction** — "15D -> 7D (53% reduction)" means you could represent this data with only 7 features instead of 15

---

### Option 7: Student Lookup & Compare

**What it does:** Compares two students side-by-side with similarity metrics.

**What it asks:**
```
Student A # (1): 1
Student B # (100): 200
```

**What to look for:**
- **Diff column** — green `+N` means Student A is higher, red `-N` means lower, `==` means equal
- **Euclidean Distance** — raw difference (lower = more similar)
- **Cosine Similarity** — direction similarity (closer to 1.0 = more similar)
- **Eigenspace Distance** — distance in the PCA-reduced 2D space

**Fun experiment:** Compare student 1 (low performer, G3=6) with student 42 (high performer, G3=18) to see a large contrast.

---

### Option 8: Generate All Charts

**What it does:** Saves 7 image files to the `plots/` folder.

**What it asks:** Nothing — just runs and shows progress:
```
[1/7] Grade distributions... OK
[2/7] Correlation heatmap... OK
...
[7/7] Prediction comparison... OK
```

**After it finishes:** Open the `plots/` folder to view the images. They're dark-themed and high-resolution.

---

## Quick Reference

| Action | What to type |
|--------|------------|
| Select a menu option | Type `1`-`8`, press Enter |
| Accept a default value | Just press Enter |
| Enter a student number | Type a number, e.g., `150` |
| Select multiple features | Comma-separated, e.g., `12,13,14` |
| Skip a field (predict it) | Just press Enter (leave blank) |
| Exit the application | Type `0`, press Enter |

## Recommended Demo Order

If you're presenting to a teacher, this order works best:

1. **Option 1** → Show the dataset (sets context)
2. **Option 2** → Run the full analysis (proves the math)
3. **Option 3** → Predict a student's grades (shows practical use)
4. **Option 4** → Ask the teacher to enter THEIR data (wow factor)
5. **Option 6** → Show PCA patterns (hidden insights)
6. **Option 8** → Generate charts (visual proof)
