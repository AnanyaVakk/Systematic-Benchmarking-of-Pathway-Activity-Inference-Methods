"""
02 - Batch-effect assessment on the TCGA-BRCA log-CPM matrix.

Aligns samples between the count matrix and the metadata, removes single-sample
Plate/TSS batches, computes log2-CPM, then shows PCA and variance partitioning
(Plate / TSS / Condition) before and after ComBat batch correction.

Inputs  (in --data-dir, default data/processed):
    merged_count_matrix_UUID.csv, TCGA_brca_Metadata.tsv
Outputs: PCA plots (shown on screen) and variance tables (printed).
         The ComBat-corrected matrix is not saved; pathway scoring (step 03)
         starts from the raw counts.

Usage:
    python 01_batch_effect_assessment.py --data-dir data/processed
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pycombat import Combat
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

parser = argparse.ArgumentParser(description="Batch-effect assessment of the TCGA-BRCA expression data.")
parser.add_argument("--data-dir", default="data/processed",
                    help="Folder with the pipeline inputs (default: data/processed).")
args = parser.parse_args()

meta = pd.read_csv(os.path.join(args.data_dir, "TCGA_brca_Metadata.tsv"), sep="\t")
counts = pd.read_csv(os.path.join(args.data_dir, "merged_count_matrix_UUID.csv"), sep="\t")
#counts.columns = [counts.columns[0]] + [f"{col}.tsv" for col in counts.columns[1:]]
#counts.to_csv("TCGA_TCGA_counts.tsv", sep="\t")

#unique_vals = {}
#for col in meta.columns:
#    unique_vals[col] = meta[col].value_counts()
#print("Unique values:")
#print(unique_vals)

# Align samples
common = list(set(counts.columns[1:]).intersection(meta["File ID"]))

counts = counts[["gene_name"] + common]
meta = meta[meta["File ID"].isin(common)].reset_index(drop=True)

# Remove single-sample batches for BOTH Plate and TSS

# Step 1: Count samples per batch
plate_counts = meta["Plate"].value_counts()
tss_counts = meta["TSS"].value_counts()

# Step 2: Identify valid batches (>1 sample)
valid_plates = plate_counts[plate_counts > 1].index
valid_tss = tss_counts[tss_counts > 1].index

# Step 3: Keep only samples belonging to BOTH valid Plate AND valid TSS
meta_filtered = meta[
    meta["Plate"].isin(valid_plates) &
    meta["TSS"].isin(valid_tss)
].copy()

# Step 4: Filter counts matrix accordingly
valid_ids = meta_filtered["File ID"]

counts_filtered = counts[
    ["gene_name"] + list(valid_ids)
]

# Step 5: Reset meta
meta_filtered = meta_filtered.reset_index(drop=True)

# Replace original data (optional but recommended)
meta = meta_filtered
counts = counts_filtered

# Check
print("After removing single-sample batches:")
print("Counts:", counts.shape)
print("Meta:", meta.shape)

print("\nRemaining Plate counts:")
print(meta["Plate"].value_counts())

print("\nRemaining TSS counts:")
print(meta["TSS"].value_counts())
# --- NORMALIZATION ---

counts_only = counts.iloc[:, 1:]
cpm = counts_only.div(counts_only.sum(axis=0), axis=1) * 1e6
log_cpm = np.log2(cpm + 1)
log_cpm.index = counts["gene_name"]
print("Log CPM shape:", log_cpm.shape)
# --- PCA BEFORE BATCH CORRECTION ---

X = log_cpm.T
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

print("\nVariance explained BEFORE correction:", pca.explained_variance_ratio_)

pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
pca_df = pd.concat([pca_df, meta.reset_index(drop=True)], axis=1)

# Plots
plt.figure(figsize=(7,6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Plate", s=60)
plt.title("PCA Before Correction — Plate")
plt.show()

plt.figure(figsize=(7,6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="TSS", s=60)
plt.title("PCA Before Correction — TSS")
plt.show()

plt.figure(figsize=(7,6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Condition", s=60)
plt.title("PCA Before Correction — Condition")
plt.show()
def variance_partition(expr_matrix, meta, top_n=2000):

    gene_var = expr_matrix.var(axis=1)
    top_genes = gene_var.sort_values(ascending=False).head(top_n).index

    expr = expr_matrix.loc[top_genes].T.copy()

    expr["Plate"] = meta["Plate"].values
    expr["TSS"] = meta["TSS"].values
    expr["Condition"] = meta["Condition"].values

    variance_results = []

    for gene in top_genes:

        df = pd.DataFrame({
            "expr": expr[gene],
            "Plate": expr["Plate"],
            "TSS": expr["TSS"],
            "Condition": expr["Condition"]
        })

        model = ols("expr ~ Plate + TSS + Condition", data=df).fit()
        anova = sm.stats.anova_lm(model, typ=2)

        variance_results.append(anova["sum_sq"])

    variance_df = pd.DataFrame(variance_results)

    mean_variance = variance_df.mean()
    variance_pct = (mean_variance / mean_variance.sum()) * 100

    return variance_pct
var_before = variance_partition(log_cpm, meta)

print("\n=== Variance BEFORE Batch Correction ===")
print(var_before)
# --- GENE FILTERING (important) ---

gene_var = log_cpm.var(axis=1)
log_cpm_filtered = log_cpm.loc[gene_var > 0.01]

expressed = (log_cpm_filtered > 1).sum(axis=1)
log_cpm_filtered = log_cpm_filtered.loc[
    expressed > 0.2 * log_cpm_filtered.shape[1]
]

print("Genes after filtering:", log_cpm_filtered.shape[0])

# --- ComBat ---

# --- ENSURE ALIGNMENT (VERY IMPORTANT) ---
meta = meta.set_index("File ID")
meta = meta.loc[log_cpm_filtered.columns]   # match order with expression

# --- Prepare data ---
Y = log_cpm_filtered.T.values  # samples x genes

# Batch variable (must be categorical/string)
batch = meta["Plate"].astype(str).values

# Biological variable (Condition)
if meta["Condition"].nunique() > 1:
    X_design = pd.get_dummies(meta["Condition"], drop_first=True).values
else:
    X_design = None  # avoid crash if only one condition

# --- Check shapes ---
print("Y shape:", Y.shape)
print("Batch length:", len(batch))
print("X_design shape:", None if X_design is None else X_design.shape)

# --- Remove NaNs (important) ---
Y = np.nan_to_num(Y)

# =========================
# CLEAN COMBAT PIPELINE
# =========================

# Step 1: Clean expression column names
log_cpm_filtered.columns = [
    str(col).replace(".tsv", "").strip()
    for col in log_cpm_filtered.columns
]

# Step 2: Ensure meta index
if "File ID" in meta.columns:
    meta = meta.set_index("File ID")

meta.index = meta.index.astype(str).str.strip()

# Step 3: REMOVE DUPLICATES (VERY IMPORTANT)
log_cpm_filtered = log_cpm_filtered.loc[:, ~log_cpm_filtered.columns.duplicated()]
meta = meta[~meta.index.duplicated(keep="first")]

# Step 4: Find common samples
common_samples = sorted(
    set(log_cpm_filtered.columns).intersection(meta.index)
)

print("Expression samples:", len(log_cpm_filtered.columns))
print("Meta samples:", len(meta))
print("Common samples:", len(common_samples))

# Step 5: Subset BOTH
log_cpm_filtered = log_cpm_filtered[common_samples]
meta = meta.loc[common_samples]

# Step 6: FORCE SAME ORDER (THIS FIXES YOUR ERROR)
meta = meta.reindex(log_cpm_filtered.columns)

# Step 7: FINAL CHECK
print("Expression sample count:", len(log_cpm_filtered.columns))
print("Meta sample count:", len(meta))

if not list(log_cpm_filtered.columns) == list(meta.index):
    print("Still mismatch — showing differences:")
    print("In expression not in meta:", set(log_cpm_filtered.columns) - set(meta.index))
    print("In meta not in expression:", set(meta.index) - set(log_cpm_filtered.columns))
    raise ValueError("Alignment failed")

print("Alignment successful!")

# =========================
# RUN COMBAT
# =========================

batch_counts = meta["Plate"].value_counts()

# Keep only batches with >=2 samples
valid_batches = batch_counts[batch_counts > 1].index

# Filter metadata
meta = meta[meta["Plate"].isin(valid_batches)]

# Filter expression to match
log_cpm_filtered = log_cpm_filtered[meta.index]

print("After re-filtering batches:")
print("Remaining samples:", len(meta))
print("Remaining batches:", meta["Plate"].nunique())

# Final check
assert list(log_cpm_filtered.columns) == list(meta.index)

#combat starts
Y = log_cpm_filtered.T.values  # samples x genes
batch = meta["Plate"].astype(str).values

# Design matrix
if meta["Condition"].nunique() > 1:
    X_design = pd.get_dummies(meta["Condition"], drop_first=True).values
else:
    X_design = None

# Debug shapes
print("Y shape:", Y.shape)
print("Batch length:", len(batch))
print("X_design shape:", None if X_design is None else X_design.shape)

# Remove NaNs
Y = np.nan_to_num(Y)

# Run ComBat
combat = Combat()

if X_design is not None:
    Y_combat = combat.fit_transform(Y, batch, X=X_design)
else:
    Y_combat = combat.fit_transform(Y, batch)

# Convert back
combat_data = pd.DataFrame(
    Y_combat.T,
    index=log_cpm_filtered.index,
    columns=log_cpm_filtered.columns
)

print("ComBat completed successfully!")
# --- PCA AFTER CORRECTION ---

X_post = combat_data.T
X_post_scaled = StandardScaler().fit_transform(X_post)

pca_post = PCA(n_components=2)
pcs_post = pca_post.fit_transform(X_post_scaled)

print("\nVariance explained AFTER correction:", pca_post.explained_variance_ratio_)

pca_post_df = pd.DataFrame(pcs_post, columns=["PC1", "PC2"])
pca_post_df = pd.concat([pca_post_df, meta.reset_index(drop=True)], axis=1)

# Plots
plt.figure(figsize=(7,6))
sns.scatterplot(data=pca_post_df, x="PC1", y="PC2", hue="Plate", s=60)
plt.title("PCA After Correction — Plate")
plt.show()

plt.figure(figsize=(7,6))
sns.scatterplot(data=pca_post_df, x="PC1", y="PC2", hue="TSS", s=60)
plt.title("PCA After Correction — TSS")
plt.show()

plt.figure(figsize=(7,6))
sns.scatterplot(data=pca_post_df, x="PC1", y="PC2", hue="Condition", s=60)
plt.title("PCA After Correction — Condition")
plt.show()
var_after = variance_partition(combat_data, meta)

print("\n=== Variance AFTER Batch Correction ===")
print(var_after)
