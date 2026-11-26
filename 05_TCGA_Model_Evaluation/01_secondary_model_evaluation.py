"""
05 - Secondary evaluation of the tuned models.

Re-evaluates each best estimator from step 04 with 10-fold stratified CV on the
full data and writes per-fold metrics, mean/SD/95% CI summaries, ROC and PR
curves, boxplots, rankings and a PR-AUC heatmap.

Inputs  : <data-dir>/FINAL_Pathway_*.csv, <results-dir>/models/best_estimators.pkl
Outputs : <results-dir>/cv_evaluation/  (CSV tables and PNG figures)

Usage:
    python 01_secondary_model_evaluation.py --data-dir data/processed --results-dir results
"""
import argparse
# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    average_precision_score, roc_auc_score,
    balanced_accuracy_score, accuracy_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix,
    brier_score_loss
)

# ==========================================
# SETTINGS
# ==========================================
parser = argparse.ArgumentParser(description="Secondary evaluation of the tuned models.")
parser.add_argument("--data-dir", default="data/processed",
                    help="Folder with FINAL_Pathway_*.csv (default: data/processed).")
parser.add_argument("--results-dir", default="results",
                    help="Folder for outputs (default: results).")
args = parser.parse_args()

DATA_DIR = args.data_dir
BEST_ESTIMATORS_PATH = os.path.join(args.results_dir, "models", "best_estimators.pkl")
OUTPUT_DIR = os.path.join(args.results_dir, "cv_evaluation")
RANDOM_STATE = 42
N_SPLITS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD BEST MODELS
# ==========================================
with open(BEST_ESTIMATORS_PATH, "rb") as f:
    best_models = pickle.load(f)

# ==========================================
# DATA LOADER
# ==========================================
def load_pathway_csv(file_path):

    df = pd.read_csv(file_path)
    df.set_index("File_ID", inplace=True)
    df = df.dropna(subset=["Condition"])

    y = df["Condition"].astype(str).str.strip().map({
        "Tumor": 1,
        "Solid_Tissue_Normal": 0
    })

    X = df.drop(columns=["Condition"])

    return X, y


# ==========================================
# FILE MAP
# ==========================================
file_paths = {
    "ssGSEA": os.path.join(DATA_DIR, "FINAL_Pathway_ssgsea.csv"),
    "GSVA": os.path.join(DATA_DIR, "FINAL_Pathway_gsva.csv"),
    "PLAGE": os.path.join(DATA_DIR, "FINAL_Pathway_plage.csv"),
    "ZScore": os.path.join(DATA_DIR, "FINAL_Pathway_zscore.csv"),
    "PROGENy": os.path.join(DATA_DIR, "FINAL_Pathway_PROGENy.csv")
}

# ==========================================
# METRICS
# ==========================================
def compute_metrics(y_true, y_prob, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "PR_AUC": average_precision_score(y_true, y_prob),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),

        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),

        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "Precision_macro": precision_score(y_true, y_pred, average="macro"),
        "Recall_macro": recall_score(y_true, y_pred, average="macro"),

        "F1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "Precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "Recall_weighted": recall_score(y_true, y_pred, average="weighted"),

        "Specificity": specificity,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob)
    }

# ==========================================
# FUNCTION: 95% CI
# ==========================================
def compute_confidence_interval(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    ci = 1.96 * (std / np.sqrt(n))
    return mean, std, mean - ci, mean + ci

# ==========================================
# MAIN LOOP
# ==========================================
all_results = []

for pathway in best_models.keys():

    print(f"\n===== {pathway} =====")

    X, y = load_pathway_csv(file_paths[pathway])

    for model_name in best_models[pathway].keys():

        print(f"\n--- Running {pathway} + {model_name} ---")

        model = best_models[pathway][model_name]

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        results = []

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        recalls_interp = np.linspace(0, 1, 100)
        precisions_interp = []

        # ======================================
        # CV LOOP
        # ======================================
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_test, y_prob, y_pred)
            metrics["Fold"] = fold

            results.append(metrics)

            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(auc(fpr, tpr))

            # PR
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            precision_interp = np.interp(recalls_interp, recall[::-1], precision[::-1])
            precisions_interp.append(precision_interp)

        # ======================================
        # SAVE FOLD RESULTS
        # ======================================
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{OUTPUT_DIR}/{pathway}_{model_name}_cv.csv", index=False)

        # ======================================
        # MEAN + STD + 95% CI
        # ======================================
        summary = {
            "Pathway": pathway,
            "Model": model_name
        }

        for col in results_df.columns:
            if col == "Fold":
                continue

            values = results_df[col].values
            mean, std, ci_low, ci_high = compute_confidence_interval(values)

            summary[f"{col}_mean"] = mean
            summary[f"{col}_std"] = std
            summary[f"{col}_CI95_low"] = ci_low
            summary[f"{col}_CI95_high"] = ci_high

        # save per-pair summary
        pd.DataFrame([summary]).to_csv(
            f"{OUTPUT_DIR}/{pathway}_{model_name}_summary.csv",
            index=False
        )

        all_results.append(summary)

        # ======================================
        # ROC PLOT
        # ======================================
        plt.figure()
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        plt.plot(mean_fpr, mean_tpr, label=f"AUC = {mean_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.title(f"ROC: {pathway}-{model_name}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()

        plt.savefig(f"{OUTPUT_DIR}/{pathway}_{model_name}_ROC.png")
        plt.close()

        # ======================================
        # PR PLOT
        # ======================================
        plt.figure()
        mean_precision = np.mean(precisions_interp, axis=0)
        mean_pr_auc = auc(recalls_interp, mean_precision)

        plt.plot(recalls_interp, mean_precision, label=f"PR AUC = {mean_pr_auc:.3f}")

        plt.title(f"PR: {pathway}-{model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        plt.savefig(f"{OUTPUT_DIR}/{pathway}_{model_name}_PR.png")
        plt.close()

        # ======================================
        # BOXPLOT
        # ======================================
        plt.figure()
        results_df.boxplot()
        plt.xticks(rotation=90)
        plt.title(f"{pathway}-{model_name}")
        plt.tight_layout()

        plt.savefig(f"{OUTPUT_DIR}/{pathway}_{model_name}_boxplot.png")
        plt.close()

# ==========================================
# FINAL SUMMARY
# ==========================================
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(f"{OUTPUT_DIR}/FINAL_SUMMARY.csv", index=False)

ranking_df = summary_df.sort_values("PR_AUC_mean", ascending=False)

ranking_df.to_csv(f"{OUTPUT_DIR}/RANKING_BY_PR_AUC.csv", index=False)

print("\n===== TOP MODELS =====")
print(ranking_df[["Pathway", "Model", "PR_AUC_mean", "PR_AUC_std"]].head(10))

best_per_pathway = (
    summary_df.sort_values("PR_AUC_mean", ascending=False)
    .groupby("Pathway")
    .first()
    .reset_index()
)

best_per_pathway.to_csv(f"{OUTPUT_DIR}/BEST_PER_PATHWAY.csv", index=False)

best_per_model = (
    summary_df.sort_values("PR_AUC_mean", ascending=False)
    .groupby("Model")
    .first()
    .reset_index()
)

best_per_model.to_csv(f"{OUTPUT_DIR}/BEST_PER_MODEL.csv", index=False)

pivot = summary_df.pivot(index="Pathway", columns="Model", values="PR_AUC_mean")

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".3f")

plt.title("PR-AUC Heatmap (Pathway vs Model)")
plt.savefig(f"{OUTPUT_DIR}/HEATMAP_PR_AUC.png")
plt.close() 
