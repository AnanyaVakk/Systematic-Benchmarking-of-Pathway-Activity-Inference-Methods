import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import argparse
import re
import os

# ============================================================
# STEP 14A — REPRODUCIBILITY LAYER  (CORRECTED VERSION)
# Based directly on Step 13 stability outputs + pathway matrices
#
# FIXES APPLIED vs the previous version:
#   1. STABILITY_DETAILS / STABILITY_SUMMARY now point at
#      "results/stability/", matching where Step 13 actually saved
#      them (confirmed from Step 13's own printed output:
#      "Saved only: results/stability/stability_summary_all_methods.csv").
#      The old version looked in the current directory and would have
#      raised FileNotFoundError.
#   2. PATHWAY_FILES now uses "ZScore" (exact spelling/capitalization),
#      matching the label Step 13 actually writes into its output
#      (confirmed directly from your uploaded
#      tumor_normal_pathway_activity_tests.csv, which has method
#      labels ['GSVA','PLAGE','PROGENy','Zscore','ssGSEA'] -- note
#      even THAT run still had a capitalization mismatch: 'Zscore'
#      with lowercase 's', not 'ZScore'). Because Python string
#      matching is case-sensitive, the previous key ("Z-score", then
#      "Zscore") never matched the pivot table's real "ZScore" column,
#      so Section 4 silently added a fake all-zero "Z-score"/"Zscore"
#      column and DROPPED the real one when reindexing. This was
#      confirmed directly against your uploaded results: 22 of 41
#      pathways in reproducibility_layer_method_matrix.csv were
#      undercounted by exactly 1 (the missing ZScore contribution)
#      versus the independently-correct reproducibility_layer_final.csv.
#
#      IMPORTANT: if your own Step 13 output uses a different exact
#      label (check the unique values printed by this script's new
#      Section 2b diagnostic below), edit PATHWAY_FILES' key to match
#      EXACTLY -- case and all.
#
# INPUTS expected:
#   results/stability/stability_pathway_details.csv
#   results/stability/stability_summary_all_methods.csv
#   FINAL_Pathway_ssgsea.csv          (current directory)
#   FINAL_Pathway_gsva.csv
#   FINAL_Pathway_plage.csv
#   FINAL_Pathway_zscore.csv
#   FINAL_Pathway_PROGENy.csv
#
# OUTPUTS:
#   results/biological_validation/
#       reproducibility_layer_final.csv
#       reproducibility_layer_method_matrix.csv
#       reproducibility_layer_stable_pathways.csv
#       reproducibility_layer_final_with_biological_difference.csv
#       tumor_normal_pathway_activity_tests.csv
#       figures/
#           reproducibility_heatmap.png
#           recurrence_barplot.png
#           stability_vs_recurrence.png
#           boxplot_<pathway>.png  (one per top pathway)
# ============================================================

# -----------------------------
# USER SETTINGS
# -----------------------------
parser = argparse.ArgumentParser(description="Biological validation / reproducibility layer.")
parser.add_argument("--data-dir", default="data/processed",
                    help="Folder with FINAL_Pathway_*.csv (default: data/processed).")
parser.add_argument("--results-dir", default="results",
                    help="Folder with stability/ and for outputs (default: results).")
args = parser.parse_args()

BASE_DIR = Path(args.data_dir)              # FINAL_Pathway_*.csv files live here
RESULTS_DIR = Path(args.results_dir)
STABILITY_DIR = RESULTS_DIR / "stability"   # written by 06_.../02_stability_selection.py

STABILITY_DETAILS = STABILITY_DIR / "stability_pathway_details.csv"
STABILITY_SUMMARY = STABILITY_DIR / "stability_summary_all_methods.csv"

# FIX: key must EXACTLY match (case-sensitive) the label Step 13 writes.
# Verify against the Section 2b diagnostic print if you're not sure.
PATHWAY_FILES = {
    "ssGSEA": BASE_DIR / "FINAL_Pathway_ssgsea.csv",
    "GSVA": BASE_DIR / "FINAL_Pathway_gsva.csv",
    "PLAGE": BASE_DIR / "FINAL_Pathway_plage.csv",
    "ZScore": BASE_DIR / "FINAL_Pathway_zscore.csv",
    "PROGENy": BASE_DIR / "FINAL_Pathway_PROGENy.csv",
}

OUTPUT_DIR = RESULTS_DIR / "biological_validation"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

STABLE_FREQ_THRESHOLD = 0.80
TOP_N_FOR_FIGURES = 20
TOP_N_BOXPLOT = 6
RANDOM_STATE = 42

# ============================================================
# HELPERS
# ============================================================

def clean_pathway_name(x):
    x = str(x).strip()
    x = re.sub(r"\s+", "_", x)
    return x


def find_column(df, candidates, required=True):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for c in df.columns:
        cl = str(c).strip().lower()
        for candidate in candidates:
            if candidate.lower() in cl:
                return c
    if required:
        raise ValueError(
            f"Could not find any of {candidates}.\n"
            f"Available columns:\n{list(df.columns)}"
        )
    return None


def safe_mann_whitney(x, y):
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    if x.nunique() < 2 and y.nunique() < 2:
        return np.nan, np.nan
    try:
        stat, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def rank_biserial_effect(x, y):
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    try:
        u, _ = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        return float((2 * u / (n1 * n2)) - 1)
    except Exception:
        return np.nan


def canonical_condition(value):
    s = str(value).strip().lower()
    if s == "tumor":
        return "Tumor"
    if s in {"solid_tissue_normal", "solid tissue normal", "normal", "solid-tissue-normal"}:
        return "Solid_Tissue_Normal"
    return np.nan


# ============================================================
# 1. LOAD STEP 13 RESULTS
# ============================================================

print("=" * 70)
print("STEP 14A — REPRODUCIBILITY LAYER (corrected)")
print("=" * 70)

if not STABILITY_DETAILS.exists():
    raise FileNotFoundError(
        f"Missing: {STABILITY_DETAILS}\n"
        f"Check that Step 13 was run from a directory where "
        f"'results/stability/' is reachable from here, or edit "
        f"STABILITY_DIR above to point at the right location."
    )
if not STABILITY_SUMMARY.exists():
    raise FileNotFoundError(f"Missing: {STABILITY_SUMMARY}")

details = pd.read_csv(STABILITY_DETAILS)
summary = pd.read_csv(STABILITY_SUMMARY)

print(f"\nLoaded Step 13 files:\n  Details : {details.shape}\n  Summary : {summary.shape}")
print(f"\nDetails columns:\n{list(details.columns)}")
print(f"\nSummary columns:\n{list(summary.columns)}")

# ============================================================
# 2. IDENTIFY STEP 13 COLUMNS
# ============================================================

pathway_col = find_column(details, ["Pathway", "pathway", "Feature"])
method_col = find_column(details, ["Method", "Pathway_Method", "pathway_method"])
model_col = find_column(details, ["Model", "ML_Model", "model"])
freq_col = find_column(details, ["Selection_Frequency", "selection_frequency",
                                  "Selection Frequency", "frequency"])

details = details.copy()
details["Pathway_Clean"] = details[pathway_col].map(clean_pathway_name)
details["Method_Clean"] = details[method_col].astype(str).str.strip()
details["Model_Clean"] = details[model_col].astype(str).str.strip()
details["Selection_Frequency"] = pd.to_numeric(details[freq_col], errors="coerce")
details = details.dropna(subset=["Pathway_Clean", "Selection_Frequency"])

# ------------------------------------------------------------
# 2b. DIAGNOSTIC (new): print the exact Method labels found, and
# cross-check them against PATHWAY_FILES so a naming mismatch is
# caught immediately and loudly -- this is exactly the class of bug
# that silently corrupted reproducibility_layer_method_matrix.csv
# last time. If you see a MISMATCH warning below, fix PATHWAY_FILES'
# key spelling to match the "Found in Step 13 output" list exactly.
# ------------------------------------------------------------
found_methods = set(details["Method_Clean"].unique())
expected_methods = set(PATHWAY_FILES.keys())

print("\n[DIAGNOSTIC] Method labels found in Step 13 output:")
print(f"  {sorted(found_methods)}")
print(f"[DIAGNOSTIC] Method labels expected (PATHWAY_FILES keys):")
print(f"  {sorted(expected_methods)}")

mismatch = found_methods.symmetric_difference(expected_methods)
if mismatch:
    print(f"\n  *** WARNING: naming mismatch detected: {sorted(mismatch)} ***")
    print(f"  This WILL cause silently wrong output (a method's data being")
    print(f"  dropped/zeroed in the method matrix) unless fixed. Edit")
    print(f"  PATHWAY_FILES above so its keys exactly match the 'found' list.")
else:
    print("  OK: method labels match exactly.")

details["Stable_Derived"] = details["Selection_Frequency"] >= STABLE_FREQ_THRESHOLD
details_stable = details[details["Stable_Derived"]].copy()

print(f"\nStep 13 stable records: {len(details_stable)}")
print(f"  Unique stable pathways: {details_stable['Pathway_Clean'].nunique()}")

# ============================================================
# 3. BUILD REPRODUCIBILITY SUMMARY
# ============================================================

method_recurrence = (details_stable.groupby("Pathway_Clean")["Method_Clean"]
                      .nunique().rename("N_Stable_Pathway_Methods"))
model_recurrence = (details_stable.groupby("Pathway_Clean")["Model_Clean"]
                     .nunique().rename("N_Stable_ML_Models"))
combination_recurrence = (details_stable.groupby("Pathway_Clean")
                           .size().rename("N_Stable_Method_Model_Combinations"))
stability_stats = (details_stable.groupby("Pathway_Clean")["Selection_Frequency"]
                    .agg(Mean_Selection_Frequency="mean",
                         Median_Selection_Frequency="median",
                         Max_Selection_Frequency="max",
                         Min_Selection_Frequency="min",
                         SD_Selection_Frequency="std"))

details_stable["Method_Model"] = details_stable["Method_Clean"] + " | " + details_stable["Model_Clean"]
method_model_recurrence = (details_stable.groupby("Pathway_Clean")["Method_Model"]
                            .nunique().rename("N_Stable_Method_Model_Records"))

repro = pd.concat(
    [method_recurrence, model_recurrence, combination_recurrence,
     method_model_recurrence, stability_stats],
    axis=1
).reset_index()

repro["N_Pathway_Inference_Methods_Total"] = len(PATHWAY_FILES)
repro["Pathway_Method_Recurrence_Fraction"] = (
    repro["N_Stable_Pathway_Methods"] / repro["N_Pathway_Inference_Methods_Total"]
)
repro = repro.sort_values(
    ["N_Stable_Pathway_Methods", "N_Stable_Method_Model_Combinations", "Mean_Selection_Frequency"],
    ascending=[False, False, False]
).reset_index(drop=True)

# ============================================================
# 4. METHOD × PATHWAY REPRODUCIBILITY MATRIX  (bug fixed here)
# ============================================================

method_matrix = (
    details_stable.assign(Value=1)
    .pivot_table(index="Pathway_Clean", columns="Method_Clean",
                 values="Value", aggfunc="max", fill_value=0)
)

# This loop is now safe: PATHWAY_FILES keys match the real Method_Clean
# labels exactly (verified by the Section 2b diagnostic above), so no
# column gets silently zeroed-and-replaced anymore.
for method in PATHWAY_FILES.keys():
    if method not in method_matrix.columns:
        print(f"  [WARN] '{method}' never appears as a stable-pathway method "
              f"in Step 13's output -- adding as an all-zero column. If this "
              f"is unexpected, re-check the diagnostic above.")
        method_matrix[method] = 0

method_matrix = method_matrix[list(PATHWAY_FILES.keys())]
method_matrix["Total_Methods"] = method_matrix.sum(axis=1)
method_matrix = method_matrix.sort_values("Total_Methods", ascending=False)
method_matrix = method_matrix.drop(columns="Total_Methods")

method_matrix.to_csv(OUTPUT_DIR / "reproducibility_layer_method_matrix.csv")

# Sanity check: cross-validate the matrix's row sums against repro's
# independently-computed N_Stable_Pathway_Methods. These MUST match --
# if they don't, something is still wrong with the column alignment.
_check = repro.merge(
    method_matrix.sum(axis=1).rename("row_sum"),
    left_on="Pathway_Clean", right_index=True, how="left"
)
_mismatches = (_check["N_Stable_Pathway_Methods"] != _check["row_sum"]).sum()
if _mismatches > 0:
    print(f"\n  *** WARNING: {_mismatches} pathway(s) have a mismatch between "
          f"the method matrix and the reproducibility table. The matrix may "
          f"still be wrong -- re-check PATHWAY_FILES naming. ***")
else:
    print(f"\n  [OK] Method matrix cross-validated against reproducibility "
          f"table: all {len(_check)} pathways match exactly.")

# ============================================================
# 5. FULL REPRODUCIBILITY TABLE
# ============================================================

repro.to_csv(OUTPUT_DIR / "reproducibility_layer_final.csv", index=False)

stable_pathways = repro[[
    "Pathway_Clean", "N_Stable_Pathway_Methods", "N_Stable_ML_Models",
    "N_Stable_Method_Model_Combinations", "Mean_Selection_Frequency",
    "Median_Selection_Frequency", "Max_Selection_Frequency",
    "Min_Selection_Frequency", "SD_Selection_Frequency",
    "Pathway_Method_Recurrence_Fraction",
]].copy()
stable_pathways.to_csv(OUTPUT_DIR / "reproducibility_layer_stable_pathways.csv", index=False)

print("\nReproducibility summary:")
print(stable_pathways.head(20).to_string(index=False))

# ============================================================
# 6. LOAD PATHWAY MATRICES
# ============================================================

pathway_data = {}

for method, path in PATHWAY_FILES.items():
    if not path.exists():
        print(f"\nWARNING: Missing pathway file: {path}")
        continue

    df = pd.read_csv(path)
    print(f"\nLoaded {method}: {df.shape}")

    sample_col = find_column(df, ["File_ID", "file_id", "Sample_ID", "sample_id"], required=False)
    condition_col = find_column(df, ["Condition", "condition", "Label", "label"], required=False)

    if sample_col is None:
        sample_col = df.columns[0]
    if condition_col is None:
        print(f"  WARNING: No Condition column detected in {method}. Skipping.")
        continue

    tmp = df.copy()
    tmp["File_ID_Clean"] = tmp[sample_col].astype(str).str.strip()
    tmp["Condition_Clean"] = tmp[condition_col].map(canonical_condition)
    tmp = tmp[tmp["Condition_Clean"].notna()].copy()

    metadata_cols = {sample_col, condition_col, "File_ID_Clean", "Condition_Clean",
                      "File_ID", "Condition", "Sample_ID", "sample_id", "Label", "label"}
    feature_cols = [c for c in tmp.columns if c not in metadata_cols]

    X = tmp[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid_features = [c for c in X.columns if X[c].notna().sum() > 0]
    X = X[valid_features]

    pathway_data[method] = {
        "sample_id": tmp["File_ID_Clean"].values,
        "condition": tmp["Condition_Clean"].values,
        "X": X,
    }
    print(f"  Samples: {len(tmp)} | Pathway features: {X.shape[1]}")

# ============================================================
# 7. TUMOR vs NORMAL BIOLOGICAL DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("TUMOR vs NORMAL PATHWAY ACTIVITY")
print("=" * 70)

diff_records = []
stable_pathway_set = set(repro["Pathway_Clean"])

for method, obj in pathway_data.items():
    sample_ids = pd.Series(obj["sample_id"])
    conditions = pd.Series(obj["condition"])
    X = obj["X"].copy()
    X.index = sample_ids.values

    tumor_mask = conditions.values == "Tumor"
    normal_mask = conditions.values == "Solid_Tissue_Normal"

    for pathway in X.columns:
        pathway_clean = clean_pathway_name(pathway)
        if pathway_clean not in stable_pathway_set:
            continue

        values = pd.to_numeric(X[pathway], errors="coerce")
        tumor_values = values.iloc[np.where(tumor_mask)[0]].dropna()
        normal_values = values.iloc[np.where(normal_mask)[0]].dropna()

        u_stat, p_value = safe_mann_whitney(tumor_values, normal_values)
        effect = rank_biserial_effect(tumor_values, normal_values)

        diff_records.append({
            "Pathway": pathway_clean,
            "Method": method,
            "N_Tumor": len(tumor_values),
            "N_Normal": len(normal_values),
            "Tumor_Median": tumor_values.median() if len(tumor_values) else np.nan,
            "Normal_Median": normal_values.median() if len(normal_values) else np.nan,
            "Median_Difference": (
                tumor_values.median() - normal_values.median()
                if len(tumor_values) and len(normal_values) else np.nan
            ),
            "Rank_Biserial_Effect": effect,
            "MW_U": u_stat,
            "MW_P": p_value,
        })

diff_df = pd.DataFrame(diff_records)

if len(diff_df) > 0:
    valid = diff_df["MW_P"].notna()
    diff_df["MW_FDR"] = np.nan
    if valid.sum() > 0:
        diff_df.loc[valid, "MW_FDR"] = multipletests(diff_df.loc[valid, "MW_P"], method="fdr_bh")[1]
    diff_df.to_csv(OUTPUT_DIR / "tumor_normal_pathway_activity_tests.csv", index=False)
    print(f"\nSaved {len(diff_df)} pathway x method Tumor-vs-Normal tests.")
else:
    print("\nNo Tumor-vs-Normal results were generated. Check pathway CSV column names.")

# ============================================================
# 8. MERGE BIOLOGICAL DIFFERENCE INTO FINAL TABLE
# ============================================================

if len(diff_df) > 0:
    bio_summary = (
        diff_df.groupby("Pathway")
        .agg(N_Methods_Tested=("Method", "nunique"),
             N_Methods_FDR_Significant=("MW_FDR", lambda x: int((x < 0.05).sum())),
             Mean_Abs_Effect=("Rank_Biserial_Effect", lambda x: np.nanmean(np.abs(x))),
             Mean_Effect=("Rank_Biserial_Effect", "mean"),
             Best_FDR=("MW_FDR", "min"),
             Median_FDR=("MW_FDR", "median"))
        .reset_index()
    )
    final_table = repro.merge(bio_summary, left_on="Pathway_Clean", right_on="Pathway", how="left") \
                        .drop(columns=["Pathway"])
else:
    final_table = repro.copy()

final_table = final_table.sort_values(
    ["N_Stable_Pathway_Methods", "N_Stable_Method_Model_Combinations", "Mean_Selection_Frequency"],
    ascending=[False, False, False]
)
final_table.to_csv(OUTPUT_DIR / "reproducibility_layer_final_with_biological_difference.csv", index=False)

# ============================================================
# 9. FIGURE 1 — REPRODUCIBILITY HEATMAP
# ============================================================

top_pathways = repro.head(TOP_N_FOR_FIGURES)["Pathway_Clean"].tolist()
heatmap_df = method_matrix.loc[method_matrix.index.intersection(top_pathways)].copy()
heatmap_df = heatmap_df.reindex(top_pathways[::-1]).dropna(how="all")

if len(heatmap_df) > 0:
    fig, ax = plt.subplots(figsize=(11, max(6, 0.38 * len(heatmap_df))))
    im = ax.imshow(heatmap_df.values, aspect="auto", interpolation="nearest" , cmap="GnBu")
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            ax.text(j, i, str(int(heatmap_df.iloc[i, j])), ha="center", va="center")
    ax.set_xlabel("Pathway activity inference method")
    ax.set_ylabel("Stable pathway")
    ax.set_title("Reproducibility of Stable Pathway Selection Across Inference Methods")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Stable in >=1 ML model")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reproducibility_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 10. FIGURE 2 — RECURRENCE BARPLOT
# ============================================================

plot_df = repro.head(TOP_N_FOR_FIGURES).copy().sort_values("N_Stable_Method_Model_Combinations")
fig, ax = plt.subplots(figsize=(10, max(6, 0.38 * len(plot_df))))
ax.barh(plot_df["Pathway_Clean"], plot_df["N_Stable_Method_Model_Combinations"])
ax.set_xlabel("Number of stable method x model combinations")
ax.set_ylabel("Pathway")
ax.set_title("Cross-Combination Reproducibility of Stable Pathways")
fig.tight_layout()
fig.savefig(FIG_DIR / "recurrence_barplot.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 11. FIGURE 3 — STABILITY VS RECURRENCE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(repro["N_Stable_Method_Model_Combinations"], repro["Mean_Selection_Frequency"], alpha=0.8)
ax.set_xlabel("Number of stable method x model combinations")
ax.set_ylabel("Mean selection frequency")
ax.set_title("Pathway Stability vs Cross-Combination Reproducibility")
label_df = repro.head(min(10, len(repro)))
for _, row in label_df.iterrows():
    ax.annotate(row["Pathway_Clean"],
                (row["N_Stable_Method_Model_Combinations"], row["Mean_Selection_Frequency"]),
                xytext=(5, 5), textcoords="offset points", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "stability_vs_recurrence.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 12. FIGURE 4 — TUMOR vs NORMAL BOXPLOTS
# ============================================================

if len(diff_df) > 0:
    box_pathways = repro.head(TOP_N_BOXPLOT)["Pathway_Clean"].tolist()
    box_records = []

    for method, obj in pathway_data.items():
        X = obj["X"].copy()
        conditions = pd.Series(obj["condition"]).reset_index(drop=True)

        for pathway in X.columns:
            pathway_clean = clean_pathway_name(pathway)
            if pathway_clean not in box_pathways:
                continue
            vals = pd.to_numeric(X[pathway], errors="coerce").reset_index(drop=True)
            tmp = pd.DataFrame({"Pathway": pathway_clean, "Method": method,
                                 "Condition": conditions, "Activity": vals})
            tmp = tmp.dropna(subset=["Activity"])
            box_records.append(tmp)

    if box_records:
        box_df = pd.concat(box_records, ignore_index=True)

        for pathway in box_pathways:
            sub = box_df[box_df["Pathway"] == pathway].copy()
            if sub.empty:
                continue
            methods_present = sub["Method"].unique()
            fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(methods_present)), 6))

            positions, data, labels = [], [], []
            pos = 1
            for method in methods_present:
                tumor = sub[(sub["Method"] == method) & (sub["Condition"] == "Tumor")]["Activity"].dropna()
                normal = sub[(sub["Method"] == method) & (sub["Condition"] == "Solid_Tissue_Normal")]["Activity"].dropna()
                if len(tumor) > 0:
                    data.append(tumor.values); positions.append(pos); labels.append(f"{method}\nTumor"); pos += 1
                if len(normal) > 0:
                    data.append(normal.values); positions.append(pos); labels.append(f"{method}\nNormal"); pos += 2

            if data:
                ax.boxplot(data, positions=positions, widths=0.65, showfliers=False)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_ylabel("Pathway activity")
                ax.set_title(f"Tumor vs Normal Activity: {pathway}")
                ax.axhline(0, linestyle="--", linewidth=0.8)
                fig.tight_layout()
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", pathway)
                fig.savefig(FIG_DIR / f"boxplot_{safe_name}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

# ============================================================
# 13. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 14A COMPLETE")
print("=" * 70)
print(f"\nOutput directory:\n{OUTPUT_DIR.resolve()}")
print("\nMain files:")
for p in sorted(OUTPUT_DIR.glob("*.csv")):
    print("  ", p.name)
print("\nFigures:")
for p in sorted(FIG_DIR.glob("*.png")):
    print("  ", p.name)
print("\nTop reproducible pathways:")
print(final_table[["Pathway_Clean", "N_Stable_Pathway_Methods", "N_Stable_ML_Models",
                    "N_Stable_Method_Model_Combinations", "Mean_Selection_Frequency"]]
      .head(15).to_string(index=False))