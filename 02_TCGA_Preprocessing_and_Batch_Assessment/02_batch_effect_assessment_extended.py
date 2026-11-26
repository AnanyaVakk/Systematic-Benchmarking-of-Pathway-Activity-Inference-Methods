"""
02 - Extended batch-effect assessment (exploratory).

Adds to 01_batch_effect_assessment.py: batch-vs-condition confounding tables,
library-size comparison between Tumor and Solid_Tissue_Normal, PCA loadings,
and ComBat correction with before/after variance partitioning.

Inputs  (in --data-dir, default data/processed):
    merged_count_matrix_UUID.csv, TCGA_brca_Metadata.tsv
Outputs: plots (shown on screen) and tables (printed).

Usage:
    python 02_batch_effect_assessment_extended.py --data-dir data/processed
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

# Batch label used to exclude the A277_44 outlier batch from the PCA plots.
# (Built here because 02_add_batch_metadata.py does not create it.)
meta["Batch_Plate_TSS"] = meta["Plate"].astype(str) + "_" + meta["TSS"].astype(str)

# Check
print("After removing single-sample batches:")
print("Counts:", counts.shape)
print("Meta:", meta.shape)

print("\nRemaining Plate counts:")
print(meta["Plate"].value_counts())

print("\nRemaining TSS counts:")
print(meta["TSS"].value_counts())

print("Counts:", counts.shape)
print("Metadata:", meta.shape)

print(meta["Condition"].value_counts())

# Batch vs. biology confounding
print(pd.crosstab(meta["Plate"], meta["Condition"]))
print(pd.crosstab(meta["TSS"], meta["Condition"]))

# Library size distribution by Condition
lib_size = counts.iloc[:, 1:].sum(axis=0)

print(len(lib_size))
print(len(meta["Condition"]))

print(lib_size.index[:5])
print(meta["File ID"][:5])

lib_df = pd.DataFrame({
    "LibrarySize": lib_size.values,
    "Condition": meta["Condition"].values
})

# plt.figure(figsize=(7,5))
# sns.boxplot(data=lib_df, x="Condition", y="LibrarySize")
# plt.title("Library Size by Condition")
# plt.show()

print(lib_df.groupby("Condition")["LibrarySize"].describe())

# Coefficient of variation
summary = lib_df.groupby("Condition")["LibrarySize"].agg(
    ["mean", "std", "median", "min", "max"]
)

summary["CV"] = summary["std"] / summary["mean"]
print(summary)

# Statistical Significance Test
tumor  = lib_df[lib_df["Condition"]=="Tumor"]["LibrarySize"]
normal = lib_df[lib_df["Condition"]=="Solid_Tissue_Normal"]["LibrarySize"]

stat, p = mannwhitneyu(tumor, normal, alternative="two-sided")

print("Mann–Whitney U p-value:", p)

# Fold Change in Sequencing Depth:
fc = tumor.mean() / normal.mean()
print("Tumor / Normal depth fold-change:", fc)


# Normalisation:
# Convert to CPM
counts_only = counts.iloc[:, 1:]

cpm = counts_only.div(counts_only.sum(axis=0), axis=1) * 1e6
log_cpm = np.log2(cpm + 1)

log_cpm.index = counts["gene_name"]

# Transpose → samples x genes
X = log_cpm.T

# Scale
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
pca_df = pd.concat([pca_df, meta.reset_index(drop=True)], axis=1)

print("Variance explained:", pca.explained_variance_ratio_)

loadings = pd.DataFrame(
    pca.components_.T,
    index=log_cpm.index,   # gene names
    columns=["PC1", "PC2"]
)

top_pc1_pos = loadings["PC2"].sort_values(ascending=False).head(10)

print("\nTop positive PC2 genes:")
print(top_pc1_pos)

top_pc1_neg = loadings["PC2"].sort_values(ascending=True).head(10)

print("\nTop negative PC2 genes:")
print(top_pc1_neg)

loadings["abs_PC2"] = loadings["PC2"].abs()

top_abs = loadings.sort_values("abs_PC2", ascending=False).head(20)

print("\nTop absolute PC2 contributors:")
print(top_abs[["PC2"]])

# PCA by Plate
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df[pca_df["Batch_Plate_TSS"] != "A277_44"],
    x="PC1",
    y="PC2",
    hue="Plate",
    palette="tab20",
    s=60
)
plt.title("PCA colored by Plate")
plt.legend(bbox_to_anchor=(1.05,1))
plt.show()

# PCA by TSS
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df[pca_df["Batch_Plate_TSS"] != "A277_44"],
    x="PC1",
    y="PC2",
    hue="TSS",
    palette="tab20",
    s=60
)
plt.title("PCA colored by TSS")
plt.legend(bbox_to_anchor=(1.05,1))
plt.show()

# PCA by Sample Type
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df[pca_df["Batch_Plate_TSS"] != "A277_44"],
    x="PC1",
    y="PC2",
    hue="Condition",
    s=60
)
plt.title("PCA colored by Condition")
plt.show()

print(pd.crosstab(
    meta["Plate"],
    meta["Condition"]
).loc["A277"]
)

# print(meta[meta["Plate"]=="A277"]["Condition"].value_counts())
# print(pd.crosstab(meta["TSS"], meta["Plate"]))

# Check variance explained by batch vs biology

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

        model = ols(
            "expr ~ Plate + TSS + Condition",
            data=df
        ).fit()

        anova = sm.stats.anova_lm(model, typ=2)

        variance_results.append(anova["sum_sq"])

    variance_df = pd.DataFrame(variance_results)

    mean_variance = variance_df.mean()

    variance_pct = (
        mean_variance /
        mean_variance.sum()
    ) * 100

    return variance_pct


# Batch Correction:

gene_var = log_cpm.var(axis=1)
log_cpm_filtered = log_cpm.loc[
    gene_var > 0.01
]

print("Genes remaining:", log_cpm_filtered.shape[0])
expressed = (log_cpm_filtered > 1).sum(axis=1)

log_cpm_filtered = log_cpm_filtered.loc[
    expressed > 0.2 * log_cpm_filtered.shape[1]
]

Y = log_cpm_filtered.T.values
b = meta["Plate"].values
X = pd.get_dummies(meta["Condition"], drop_first=True).values
C_tss = pd.get_dummies(
    meta["TSS"],
    drop_first=True
).values

combat = Combat()

Y_combat = combat.fit_transform(Y, b, X=X)

combat_data = pd.DataFrame(
    Y_combat.T,
    index=log_cpm_filtered.index,
    columns=log_cpm_filtered.columns
)

X_post = combat_data.T

X_post_scaled = StandardScaler().fit_transform(X_post)

pca_post = PCA(n_components=2)
pcs_post = pca_post.fit_transform(X_post_scaled)

print("Variance explained after ComBat:",
      pca_post.explained_variance_ratio_)

pca_post_df = pd.DataFrame(
    pcs_post,
    columns=["PC1", "PC2"]
)

pca_post_df = pd.concat(
    [pca_post_df, meta.reset_index(drop=True)],
    axis=1
)

loadings = pd.DataFrame(
    pca.components_.T,
    index=log_cpm.index,   # gene names
    columns=["PC1", "PC2"]
)

top_pc1_pos = loadings["PC1"].sort_values(ascending=False).head(20)

print("\nTop positive PC1 genes:")
print(top_pc1_pos)

top_pc1_neg = loadings["PC1"].sort_values(ascending=True).head(20)

print("\nTop negative PC1 genes:")
print(top_pc1_neg)

loadings["abs_PC1"] = loadings["PC1"].abs()

top_abs = loadings.sort_values("abs_PC1", ascending=False).head(40)

print("\nTop absolute PC1 contributors:")
print(top_abs[["PC1"]])

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=pca_post_df,
    x="PC1",
    y="PC2",
    hue="TSS",
    palette="tab20",
    s=60
)
plt.title("PCA After ComBat — TSS")
plt.show()

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=pca_post_df,
    x="PC1",
    y="PC2",
    hue="Plate",
    palette="tab20",
    s=60
)
plt.title("PCA After ComBat — Plate")
plt.show()

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=pca_post_df,
    x="PC1",
    y="PC2",
    hue="Condition",
    s=70
)
plt.title("PCA After ComBat — Condition")
plt.show()

var_before = variance_partition(
    log_cpm,
    meta
)
print("Before Batch Correction:")
print(var_before)

var_plate = variance_partition(
    combat_data,
    meta
)
print("After Correction")
print(var_plate)
