# =============================================================================
#  HEART DISEASE PREDICTION — EDA & DATA PREPROCESSING
#  Dataset : UCI Heart Disease – Cleveland (303 instances, 14 features)
#  Libraries: pandas, numpy, matplotlib  (no seaborn / sklearn required)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
import warnings, os
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive PNG output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi"      : 120,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9,
    "figure.facecolor": "white",
    "axes.spines.top" : False,
    "axes.spines.right": False,
})

# Colour palette (tab10)
COLS = plt.cm.tab10.colors

OUT_DIR  = r"d:\Dm project\Heart Disease Prediction\Notebook"
DATA_DIR = r"d:\Dm project\Heart Disease Prediction\heart+disease"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def subsection(title):
    print(f"\n--- {title} ---")

def save_fig(name):
    path = f"{OUT_DIR}/{name}"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {name}")

# manual StandardScaler (no sklearn needed)
def standard_scale(df, cols):
    df2 = df.copy()
    for c in cols:
        mu, sigma = df2[c].mean(), df2[c].std()
        df2[c] = (df2[c] - mu) / sigma if sigma != 0 else 0.0
    return df2

# manual IQR bounds
def iqr_bounds(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Pearson correlation matrix (manual, to avoid scipy dependency)
def pearson_corr(df):
    return df.corr(method="pearson")


# =============================================================================
# 1. LOAD DATA
# =============================================================================
section("1. LOADING THE DATASET")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

df_raw = pd.read_csv(
    f"{DATA_DIR}/processed.cleveland.data",
    header=None,
    names=COLUMNS,
    na_values="?"          # '?' encodes missing values
)

print(f"\nShape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print("\nFirst 10 rows:")
print(df_raw.head(10).to_string())


# =============================================================================
# 2. DATASET OVERVIEW & STATISTICAL SUMMARY
# =============================================================================
section("2. DATASET OVERVIEW & STATISTICAL SUMMARY")

subsection("Data types")
print(df_raw.dtypes.to_string())

subsection("Statistical Summary (all features)")
print(df_raw.describe().round(2).to_string())

subsection("Missing values BEFORE preprocessing")
miss = df_raw.isnull().sum()
miss_pct = (miss / len(df_raw) * 100).round(2)
print(pd.DataFrame({"Count": miss, "Percent %": miss_pct}).to_string())

subsection("Duplicate rows BEFORE preprocessing")
print(f"  Total duplicate rows: {df_raw.duplicated().sum()}")


# =============================================================================
# 3. FEATURE DEFINITIONS
# =============================================================================
section("3. FEATURE DEFINITIONS")

feature_info = [
    ("age",      "Continuous",  "Age in years"),
    ("sex",      "Binary",      "Sex (1=Male, 0=Female)"),
    ("cp",       "Categorical", "Chest pain type (1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic)"),
    ("trestbps", "Continuous",  "Resting blood pressure (mm Hg)"),
    ("chol",     "Continuous",  "Serum cholesterol (mg/dl)"),
    ("fbs",      "Binary",      "Fasting blood sugar > 120 mg/dl (1=True)"),
    ("restecg",  "Categorical", "Resting ECG (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy)"),
    ("thalach",  "Continuous",  "Maximum heart rate achieved"),
    ("exang",    "Binary",      "Exercise induced angina (1=Yes)"),
    ("oldpeak",  "Continuous",  "ST depression induced by exercise vs rest"),
    ("slope",    "Categorical", "Slope of peak exercise ST segment (1=Up, 2=Flat, 3=Down)"),
    ("ca",       "Ordinal",     "Number of major vessels coloured by fluoroscopy (0–3)"),
    ("thal",     "Categorical", "Thalassemia (3=Normal, 6=Fixed defect, 7=Reversible defect)"),
    ("target",   "Target",      "Heart disease presence (0=No, 1-4=Yes)"),
]

print(f"\n{'Feature':<12} {'Type':<14} Description")
print("-" * 72)
for feat, ftype, desc in feature_info:
    print(f"{feat:<12} {ftype:<14} {desc}")

NUMERICAL   = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET      = "target"


# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# =============================================================================
section("4. EXPLORATORY DATA ANALYSIS")

# ─── 4a. Target variable ──────────────────────────────────────────────────────
subsection("4a. Target Variable Distribution")

tgt_raw = df_raw[TARGET].value_counts().sort_index()
df_raw["target_bin"] = (df_raw[TARGET] > 0).astype(int)
tgt_bin  = df_raw["target_bin"].value_counts().sort_index()
bin_lbl  = ["No Disease (0)", "Disease (1–4)"]

print("\nRaw class distribution:")
print(tgt_raw.rename({i: f"Class {i}" for i in tgt_raw.index}).to_string())
print("\nBinary distribution:")
for k, v in zip(bin_lbl, tgt_bin.values):
    print(f"  {k}: {v}  ({v/len(df_raw)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# bar – raw classes
axes[0].bar([str(i) for i in tgt_raw.index], tgt_raw.values,
            color=list(COLS[:len(tgt_raw)]), edgecolor="white")
axes[0].set_title("Target: Raw Class Distribution")
axes[0].set_xlabel("Heart Disease Level (0=None)")
axes[0].set_ylabel("Count")
for i, v in enumerate(tgt_raw.values):
    axes[0].text(i, v + 1.5, str(v), ha="center", fontweight="bold", fontsize=10)

# pie – binary
wedges, texts, autotexts = axes[1].pie(
    tgt_bin.values, labels=bin_lbl, autopct="%1.1f%%",
    colors=["#2196F3", "#F44336"], startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Target: Binary Distribution")

save_fig("fig_01_target_distribution.png")

# ─── 4b. Numerical feature distributions ─────────────────────────────────────
subsection("4b. Numerical Feature Distributions")

# Histograms
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(NUMERICAL):
    data = df_raw[col].dropna()
    axes[i].hist(data, bins=25, color=COLS[i], edgecolor="white", alpha=0.85)
    axes[i].axvline(data.mean(),   color="red",    linestyle="--", lw=1.5, label=f"Mean {data.mean():.1f}")
    axes[i].axvline(data.median(), color="orange", linestyle=":",  lw=1.5, label=f"Median {data.median():.1f}")
    axes[i].set_title(f"{col}")
    axes[i].set_xlabel(col); axes[i].set_ylabel("Frequency")
    axes[i].legend(fontsize=8)
axes[-1].axis("off")
fig.suptitle("Numerical Features – Histograms (Mean=red, Median=orange)", fontsize=14, fontweight="bold")
save_fig("fig_02_numerical_histograms.png")

# Distribution by target
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(NUMERICAL):
    nd = df_raw[df_raw["target_bin"] == 0][col].dropna()
    dd = df_raw[df_raw["target_bin"] == 1][col].dropna()
    axes[i].hist(nd, bins=20, density=True, alpha=0.55, color="#2196F3", label="No Disease", edgecolor="white")
    axes[i].hist(dd, bins=20, density=True, alpha=0.55, color="#F44336", label="Disease",    edgecolor="white")
    axes[i].set_title(f"{col} by Target"); axes[i].set_xlabel(col)
    axes[i].legend(fontsize=8)
axes[-1].axis("off")
fig.suptitle("Numerical Features – Distribution by Target", fontsize=14, fontweight="bold")
save_fig("fig_03_numerical_by_target.png")

# Box plots
fig, axes = plt.subplots(1, 5, figsize=(20, 6))
for i, col in enumerate(NUMERICAL):
    bp = axes[i].boxplot(df_raw[col].dropna(), patch_artist=True, notch=False,
                         medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(COLS[i])
    bp["boxes"][0].set_alpha(0.7)
    axes[i].set_title(col); axes[i].set_ylabel("Value")
fig.suptitle("Numerical Features – Box Plots", fontsize=14, fontweight="bold")
save_fig("fig_04_numerical_boxplots.png")

# Skewness & kurtosis
print("\n  Skewness & Kurtosis:")
print(f"  {'Feature':<12} {'Mean':>8} {'Std':>8} {'Skewness':>10} {'Kurtosis':>10}")
print("  " + "-" * 52)
for col in NUMERICAL:
    d = df_raw[col].dropna()
    sk = float(d.skew())
    ku = float(d.kurt())
    print(f"  {col:<12} {d.mean():>8.2f} {d.std():>8.2f} {sk:>10.3f} {ku:>10.3f}")

# ─── 4c. Categorical features ─────────────────────────────────────────────────
subsection("4c. Categorical Feature Analysis")

label_maps = {
    "sex"    : {0: "Female", 1: "Male"},
    "cp"     : {1: "Typical\nAngina", 2: "Atypical\nAngina", 3: "Non-anginal", 4: "Asymptomatic"},
    "fbs"    : {0: "FBS≤120", 1: "FBS>120"},
    "restecg": {0: "Normal", 1: "ST-T\nAbnormal", 2: "LV\nHypertrophy"},
    "exang"  : {0: "No", 1: "Yes"},
    "slope"  : {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca"     : {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal"   : {3: "Normal", 6: "Fixed Defect", 7: "Reversible"},
}

fig, axes = plt.subplots(2, 4, figsize=(22, 12))
axes = axes.flatten()
for i, col in enumerate(CATEGORICAL):
    vc = df_raw[col].value_counts().sort_index()
    bars = axes[i].bar([str(int(k)) for k in vc.index], vc.values,
                       color=list(COLS[:len(vc)]), edgecolor="white", alpha=0.85)
    axes[i].set_title(f"{col} – Value Counts")
    axes[i].set_xlabel(col); axes[i].set_ylabel("Count")
    for bar, cnt in zip(bars, vc.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(cnt), ha="center", fontsize=9, fontweight="bold")
    if col in label_maps:
        tks = sorted(label_maps[col].keys())
        axes[i].set_xticklabels([label_maps[col][k] for k in tks], rotation=25, ha="right", fontsize=9)
fig.suptitle("Categorical Features – Count Distributions", fontsize=14, fontweight="bold")
save_fig("fig_05_categorical_counts.png")

# Stacked % bar vs target
fig, axes = plt.subplots(2, 4, figsize=(22, 12))
axes = axes.flatten()
for i, col in enumerate(CATEGORICAL):
    ct = pd.crosstab(df_raw[col], df_raw["target_bin"], normalize="index") * 100
    ct.columns = ["No Disease %", "Disease %"]
    b1 = axes[i].bar([str(int(k)) for k in ct.index], ct["No Disease %"],
                     color="#2196F3", label="No Disease", alpha=0.85, edgecolor="white")
    b2 = axes[i].bar([str(int(k)) for k in ct.index], ct["Disease %"],
                     bottom=ct["No Disease %"],
                     color="#F44336", label="Disease", alpha=0.85, edgecolor="white")
    axes[i].set_title(f"{col} vs Heart Disease (%)")
    axes[i].set_xlabel(col); axes[i].set_ylabel("% within category")
    axes[i].set_ylim(0, 115)
    axes[i].legend(fontsize=8)
    if col in label_maps:
        tks = sorted(label_maps[col].keys())
        axes[i].set_xticklabels([label_maps[col][k] for k in tks], rotation=25, ha="right", fontsize=9)
fig.suptitle("Categorical Features vs Target (stacked %)", fontsize=14, fontweight="bold")
save_fig("fig_06_categorical_vs_target.png")

# ─── 4d. Correlation analysis ─────────────────────────────────────────────────
subsection("4d. Correlation Analysis")

num_cols_corr = NUMERICAL + ["target_bin"]
df_c = df_raw[num_cols_corr].apply(pd.to_numeric, errors="coerce").dropna()
corr = pearson_corr(df_c)

print("\nPearson Correlation Matrix:")
print(corr.round(3).to_string())

# Heatmap (manual, no seaborn)
fig, ax = plt.subplots(figsize=(8, 7))
n = len(corr.columns)
cmap = plt.cm.RdBu_r
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(corr.columns, fontsize=10)
for i in range(n):
    for j in range(n):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
ax.set_title("Correlation Heatmap – Numerical Features + Binary Target", fontsize=12, pad=12)
save_fig("fig_07_correlation_heatmap.png")

# Feature-target correlation bar
all_num_cats = NUMERICAL + CATEGORICAL
df_corr_all = df_raw[all_num_cats + ["target_bin"]].apply(pd.to_numeric, errors="coerce")
tgt_corr = df_corr_all.corr()["target_bin"].drop("target_bin").sort_values()

print("\nFeature correlations with binary target (sorted):")
print(tgt_corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = ["#F44336" if v > 0 else "#2196F3" for v in tgt_corr.values]
bars = ax.barh(tgt_corr.index, tgt_corr.values, color=bar_colors, alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlation with Binary Target (Pearson r)", fontsize=13)
ax.set_xlabel("Pearson Correlation Coefficient")
for bar, val in zip(bars, tgt_corr.values):
    xpos = val + 0.01 if val >= 0 else val - 0.01
    ha   = "left"      if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", ha=ha, fontsize=9)
red_p  = mpatches.Patch(color="#F44336", label="Positive correlation")
blue_p = mpatches.Patch(color="#2196F3", label="Negative correlation")
ax.legend(handles=[red_p, blue_p], fontsize=9)
save_fig("fig_08_feature_target_corr.png")

# Scatter pair plots (5×5 numerical)
fig, axes = plt.subplots(5, 5, figsize=(18, 18))
df_pair = df_raw[NUMERICAL + ["target_bin"]].dropna()
c_map   = {0: "#2196F3", 1: "#F44336"}
pt_col  = df_pair["target_bin"].map(c_map)
for i, ci in enumerate(NUMERICAL):
    for j, cj in enumerate(NUMERICAL):
        ax = axes[i][j]
        if i == j:
            ax.hist(df_pair[ci], bins=20, color="#607D8B", alpha=0.75, edgecolor="white")
        else:
            ax.scatter(df_pair[cj], df_pair[ci], c=pt_col, alpha=0.4, s=14)
        ax.tick_params(labelsize=7)
        if i == 4: ax.set_xlabel(cj, fontsize=9)
        if j == 0: ax.set_ylabel(ci, fontsize=9)
blue_p2 = mpatches.Patch(color="#2196F3", label="No Disease")
red_p2  = mpatches.Patch(color="#F44336", label="Disease")
fig.legend(handles=[blue_p2, red_p2], loc="upper right", fontsize=11)
fig.suptitle("Pair Plot – Numerical Features (coloured by target)", fontsize=14, fontweight="bold")
save_fig("fig_09_pair_plot.png")


# =============================================================================
# 5. DATA PREPROCESSING
# =============================================================================
section("5. DATA PREPROCESSING")

df = df_raw.drop(columns=["target_bin"]).copy()

# ─── 5a. Binarise target ──────────────────────────────────────────────────────
subsection("5a. Convert Multi-class Target → Binary")
print(f"\n  Original target classes : {sorted(df[TARGET].dropna().unique())}")
df[TARGET] = (df[TARGET] > 0).astype(int)
print(f"  Binarised target classes: {sorted(df[TARGET].unique())}")
val_counts = df[TARGET].value_counts().sort_index()
print(f"  Class distribution: {val_counts.to_dict()}")

# ─── 5b. Coerce all columns to numeric ───────────────────────────────────────
subsection("5b. Coerce Columns to Numeric (force '?' → NaN)")
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("\n  dtypes after coercion:")
print(df.dtypes.to_string())

# ─── 5c. Missing value imputation ────────────────────────────────────────────
subsection("5c. Missing Value Detection & Imputation")

before_miss = df.isnull().sum()
print("\n  BEFORE imputation:")
print(pd.DataFrame({
    "Missing Count": before_miss,
    "Missing %"    : (before_miss / len(df) * 100).round(2)
}).to_string())

print("\n  Strategy: median for numerical | mode for categorical")
for col in NUMERICAL:
    med = df[col].median()
    n   = df[col].isnull().sum()
    if n > 0:
        df[col] = df[col].fillna(med)
        print(f"    [{col}] filled {n} NaN → median = {med:.2f}")

for col in CATEGORICAL:
    mod = df[col].mode()[0]
    n   = df[col].isnull().sum()
    if n > 0:
        df[col] = df[col].fillna(mod)
        print(f"    [{col}] filled {n} NaN → mode = {mod}")

after_miss = df.isnull().sum()
print("\n  AFTER imputation – remaining NaN:")
remaining = after_miss[after_miss > 0]
print(remaining.to_string() if len(remaining) else "  → 0 missing values ✓")

# visualise missing before / after
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bm = before_miss[before_miss > 0]
if len(bm):
    axes[0].bar(bm.index, bm.values, color="#F44336", alpha=0.85, edgecolor="white")
    for i, v in enumerate(bm.values):
        axes[0].text(i, v + 0.05, str(v), ha="center", fontweight="bold")
else:
    axes[0].text(0.5, 0.5, "No missing values before", ha="center", va="center", transform=axes[0].transAxes)
axes[0].set_title("Missing Values BEFORE Imputation")
axes[0].set_ylabel("Count")

axes[1].bar(df.columns, after_miss.values, color="#4CAF50", alpha=0.85, edgecolor="white")
axes[1].set_title("Missing Values AFTER Imputation")
axes[1].set_ylabel("Count")
axes[1].set_xticklabels(df.columns, rotation=45, ha="right")
fig.suptitle("Missing Values: Before vs After Imputation", fontsize=13, fontweight="bold")
save_fig("fig_10_missing_values.png")

# ─── 5d. Remove duplicates ────────────────────────────────────────────────────
subsection("5d. Duplicate Record Removal")
n_before = len(df)
n_dup    = df.duplicated().sum()
print(f"\n  Duplicate rows found: {n_dup}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"  Rows BEFORE: {n_before}  |  Rows AFTER: {len(df)}  |  Removed: {n_before - len(df)}")

# ─── 5e. Outlier detection & capping (IQR) ───────────────────────────────────
subsection("5e. Outlier Detection & Capping (IQR Winsorisation)")

print(f"\n  {'Feature':<12} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Lower':>10} {'Upper':>10} {'Outliers':>9}")
print("  " + "-" * 65)

out_summary = {}
for col in NUMERICAL:
    Q1, Q3  = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR     = Q3 - Q1
    lb, ub  = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out   = int(((df[col] < lb) | (df[col] > ub)).sum())
    out_summary[col] = (lb, ub, n_out)
    print(f"  {col:<12} {Q1:>8.2f} {Q3:>8.2f} {IQR:>8.2f} {lb:>10.2f} {ub:>10.2f} {n_out:>9}")

# Cap outliers
df_before_cap = df[NUMERICAL].copy()
for col in NUMERICAL:
    lb, ub, _ = out_summary[col]
    df[col] = df[col].clip(lower=lb, upper=ub)

total_out = sum(v[2] for v in out_summary.values())
print(f"\n  Total outlier values capped: {total_out}")
print("  Method: IQR Winsorisation (clip to [Q1-1.5×IQR, Q3+1.5×IQR]) ✓")

# Box plots before / after capping
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
for i, col in enumerate(NUMERICAL):
    bp0 = axes[0][i].boxplot(df_before_cap[col], patch_artist=True,
                              medianprops=dict(color="black", lw=2))
    bp0["boxes"][0].set_facecolor("#F44336"); bp0["boxes"][0].set_alpha(0.6)
    axes[0][i].set_title(f"{col}\n(Before Capping)")

    bp1 = axes[1][i].boxplot(df[col], patch_artist=True,
                              medianprops=dict(color="black", lw=2))
    bp1["boxes"][0].set_facecolor("#4CAF50"); bp1["boxes"][0].set_alpha(0.6)
    axes[1][i].set_title(f"{col}\n(After Capping)")

fig.suptitle("Outlier Capping – Before (red) vs After (green) [IQR Winsorisation]",
             fontsize=13, fontweight="bold")
save_fig("fig_11_outlier_boxplots.png")

# ─── 5f. Encode categorical features ─────────────────────────────────────────
subsection("5f. Categorical Feature Encoding")

# Binary & ordinal features are already numeric integers; just cast them
BINARY_FEATS  = ["sex", "fbs", "exang"]
ORDINAL_FEATS = ["cp", "restecg", "slope", "ca"]
NOMINAL_FEATS = ["thal"]   # values 3, 6, 7 → one-hot

print(f"\n  Binary  features {BINARY_FEATS} → cast to int (already 0/1)")
print(f"  Ordinal features {ORDINAL_FEATS} → cast to int (meaningful order)")
print(f"  Nominal features {NOMINAL_FEATS} → one-hot encoded")

for col in BINARY_FEATS + ORDINAL_FEATS:
    df[col] = df[col].astype(int)

print(f"\n  Unique values of 'thal' before encoding: {sorted(df['thal'].unique())}")

df = pd.get_dummies(df, columns=NOMINAL_FEATS, prefix=NOMINAL_FEATS, dtype=int)
thal_cols = [c for c in df.columns if c.startswith("thal_")]
print(f"  One-hot columns created: {thal_cols}")
print(f"  Dataset shape after encoding: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ─── 5g. Feature scaling ──────────────────────────────────────────────────────
subsection("5g. Feature Scaling (Z-score / Standard Scaling)")

print("\n  Numerical features BEFORE scaling:")
print(df[NUMERICAL].describe().round(2).to_string())

df_scaled   = standard_scale(df, NUMERICAL)
df_unscaled = df.copy()   # keep unscaled for saving

print("\n  Numerical features AFTER scaling (mean≈0, std≈1):")
print(df_scaled[NUMERICAL].describe().round(3).to_string())


# =============================================================================
# 6. BEFORE / AFTER COMPARISON SUMMARY
# =============================================================================
section("6. BEFORE vs AFTER PREPROCESSING – SUMMARY TABLE")

print(f"\n  {'Metric':<45} {'Before':>10} {'After':>12}")
print("  " + "-" * 69)
print(f"  {'Shape (rows × cols)':<45} {str(df_raw.shape):>10} {str(df_unscaled.shape):>12}")
print(f"  {'Total missing values':<45} {int(df_raw.isnull().sum().sum()):>10} {int(df_unscaled.isnull().sum().sum()):>12}")
print(f"  {'Duplicate rows':<45} {int(df_raw.duplicated().sum()):>10} {int(df_unscaled.duplicated().sum()):>12}")
print(f"  {'Outliers in numerical cols (IQR)':<45} {total_out:>10} {'0 (capped)':>12}")
print(f"  {'Target binarised (0/1)':<45} {'No':>10} {'Yes':>12}")
print(f"  {"'thal' one-hot encoded":<45} {'No':>10} {'Yes':>12}")
print(f"  {'Scaling applied (Z-score)':<45} {'No':>10} {'df_scaled':>12}")


# =============================================================================
# 7. SAVE CLEANED DATASET
# =============================================================================
section("7. SAVING CLEANED DATASETS")

out_path    = f"{OUT_DIR}/heart_disease_cleaned.csv"
scaled_path = f"{OUT_DIR}/heart_disease_cleaned_scaled.csv"

df_unscaled.to_csv(out_path,    index=False)
df_scaled.to_csv(scaled_path,   index=False)

print(f"\n  [SAVED] Cleaned (unscaled): {out_path}")
print(f"          Shape: {df_unscaled.shape}")
print(f"\n  [SAVED] Cleaned (scaled)  : {scaled_path}")
print(f"          Shape: {df_scaled.shape}")

print(f"\n  Columns in cleaned dataset: {list(df_unscaled.columns)}")
print(f"\n  First 5 rows:")
print(df_unscaled.head().to_string())


# =============================================================================
# 8. FINAL CLEANED DATASET SUMMARY
# =============================================================================
section("8. FINAL CLEANED DATASET – STATISTICAL SUMMARY")

print("\nData types:")
print(df_unscaled.dtypes.to_string())
print("\nStatistical summary:")
print(df_unscaled.describe().round(3).to_string())

print(f"\n  Target distribution in cleaned data:")
tc = df_unscaled[TARGET].value_counts().sort_index()
for k, v in tc.items():
    print(f"    {k} ({'No Disease' if k == 0 else 'Disease'}): {v}  ({v/len(df_unscaled)*100:.1f}%)")


# =============================================================================
# 9. FIGURES SUMMARY
# =============================================================================
section("9. GENERATED FIGURES SUMMARY")
figures = [
    "fig_01_target_distribution.png",
    "fig_02_numerical_histograms.png",
    "fig_03_numerical_by_target.png",
    "fig_04_numerical_boxplots.png",
    "fig_05_categorical_counts.png",
    "fig_06_categorical_vs_target.png",
    "fig_07_correlation_heatmap.png",
    "fig_08_feature_target_corr.png",
    "fig_09_pair_plot.png",
    "fig_10_missing_values.png",
    "fig_11_outlier_boxplots.png",
]
for f in figures:
    full = os.path.join(OUT_DIR, f)
    status = "✓" if os.path.exists(full) else "✗ MISSING"
    print(f"  {status}  {f}")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — heart_disease_cleaned.csv saved successfully!")
print("=" * 70)
