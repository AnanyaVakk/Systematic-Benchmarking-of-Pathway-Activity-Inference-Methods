"""GSE58135 external validation using the EXISTING TCGA estimators.

This script does not train, tune, refit, recalibrate, or alter TCGA models.
It only aligns GEO pathway features to the saved TCGA feature schema and
calculates external-validation metrics.
"""

from pathlib import Path
import json
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix,
    matthews_corrcoef,
)

# ------------------------------ PATHS ------------------------------
# Keep config.py in the same directory as this script.
from config import TCGA_DIR, GEO_DIR, FROZEN_MODELS_FILE, FEATURE_SCHEMA_FILE, INPUT_DIR, PATHWAY_DIR, RESULT_DIR

MODEL_FILE = FROZEN_MODELS_FILE  # existing TCGA artifact; read-only
SCHEMA_FILE = FEATURE_SCHEMA_FILE
METADATA_FILE = INPUT_DIR / "GSE58135_metadata_135samples.csv"
SCORE_DIR = PATHWAY_DIR
OUTPUT_DIR = RESULT_DIR

SCORE_FILES = {
    "ssgsea": "GEO_Pathway_ssgsea.csv",
    "gsva": "GEO_Pathway_gsva.csv",
    "plage": "GEO_Pathway_plage.csv",
    "zscore": "GEO_Pathway_zscore.csv",
    "progeny": "GEO_Pathway_PROGENy.csv",
}

POSITIVE_LABEL = 1  # Tumor=1, Normal=0
THRESHOLD = 0.50
EXPECTED_N = 135
EXPECTED_TUMOR = 84
EXPECTED_NORMAL = 51

# PLAGE has cohort-specific SVD sign ambiguity. Keep True to reproduce the
# previous 20-model evaluation, but interpret PLAGE metrics as provisional.
EVALUATE_PLAGE = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def canonical_method(value):
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    for method in SCORE_FILES:
        if method in key:
            return method
    return None


def canonical_model_name(value):
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "elasticnet": "ElasticNet",
        "elastic": "ElasticNet",
        "rf": "RandomForest",
        "randomforest": "RandomForest",
        "svmrbf": "SVM_RBF",
        "svm": "SVM_RBF",
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
    }
    return aliases.get(key, str(value))


def load_schema(path):
    if not path.is_file():
        raise FileNotFoundError(f"TCGA feature schema not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    schema = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                method = canonical_method(key)
                if method and isinstance(value, list):
                    schema[method] = [str(x) for x in value]
                elif isinstance(value, dict):
                    walk(value)

    walk(obj)
    if not schema:
        raise ValueError(f"No method-to-feature lists found in schema: {path}")
    for method, features in schema.items():
        if len(features) != len(set(features)):
            raise ValueError(f"Duplicate features in {method} schema")
    return schema


def load_models(path):
    if not path.is_file():
        raise FileNotFoundError(f"Existing TCGA model file not found: {path}")
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError("Expected best_estimators.pkl to contain a nested dictionary.")

    models = []
    for saved_method, group in obj.items():
        method = canonical_method(saved_method)
        if method is None:
            warnings.warn(f"Unknown pathway key in model pickle; skipping: {saved_method}")
            continue
        if not isinstance(group, dict):
            warnings.warn(f"Unexpected model group for {saved_method}; skipping")
            continue
        for saved_model, estimator in group.items():
            if estimator is None:
                continue
            if not hasattr(estimator, "predict"):
                warnings.warn(f"Skipping {saved_method}/{saved_model}: object has no predict()")
                continue
            models.append((method, canonical_model_name(saved_model), str(saved_method), estimator))
    if not models:
        raise ValueError("No usable estimators found in the existing best_estimators.pkl")
    return models


def load_metadata(path):
    if not path.is_file():
        raise FileNotFoundError(f"GEO metadata not found: {path}")
    meta = pd.read_csv(path)
    if {"GSM_ID", "Condition"}.issubset(meta.columns):
        meta = meta.rename(columns={"GSM_ID": "sample_id", "Condition": "condition"})
    elif not {"sample_id", "condition"}.issubset(meta.columns):
        raise ValueError("Metadata must contain GSM_ID/Condition or sample_id/condition")

    meta["sample_id"] = meta["sample_id"].astype(str).str.strip()
    raw = meta["condition"].astype(str).str.strip().str.lower()
    label_map = {
        "tumor": 1, "tumour": 1,
        "normal": 0, "solid_tissue_normal": 0,
    }
    meta["y"] = raw.map(label_map)
    if meta["y"].isna().any():
        raise ValueError(f"Unrecognized GEO condition labels: {sorted(raw[meta['y'].isna()].unique())}")
    if meta["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in GEO metadata")

    if len(meta) != EXPECTED_N or int((meta.y == 1).sum()) != EXPECTED_TUMOR or int((meta.y == 0).sum()) != EXPECTED_NORMAL:
        raise ValueError(
            f"Expected {EXPECTED_N} samples ({EXPECTED_TUMOR} Tumor, {EXPECTED_NORMAL} Normal); "
            f"found {len(meta)} ({int((meta.y == 1).sum())} Tumor, {int((meta.y == 0).sum())} Normal)."
        )
    return meta.set_index("sample_id")


def load_scores(path, method):
    if not path.is_file():
        raise FileNotFoundError(f"GEO pathway score file not found: {path}")
    df = pd.read_csv(path)
    if "File_ID" in df.columns:
        df = df.rename(columns={"File_ID": "sample_id"})
    elif "sample_id" not in df.columns:
        first = df.columns[0]
        if df[first].astype(str).str.startswith("GSM").mean() >= 0.8:
            df = df.rename(columns={first: "sample_id"})
        else:
            raise ValueError(f"Cannot identify sample-ID column in {path.name}")

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    if df["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path.name}")
    df = df.set_index("sample_id")
    df.columns = [str(c).strip() for c in df.columns]

    # Known spelling mismatch seen in the previous GEO PROGENy output.
    if method == "progeny":
        rename = {"JAK-STAT": "JAK.STAT"}
        df = df.rename(columns=rename)
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate pathway columns in {path.name}")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def positive_probability(estimator, X):
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)
        classes = None
        if hasattr(estimator, "classes_"):
            classes = list(estimator.classes_)
        elif hasattr(estimator, "named_steps"):
            for step in reversed(list(estimator.named_steps.values())):
                if hasattr(step, "classes_"):
                    classes = list(step.classes_)
                    break
        if classes is None or POSITIVE_LABEL not in classes:
            raise ValueError(f"Cannot identify positive class {POSITIVE_LABEL} in estimator classes={classes}")
        return np.asarray(probs)[:, classes.index(POSITIVE_LABEL)]

    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X)).reshape(-1)
        # Only valid for conventional binary classifiers with classes ordered [0, 1].
        classes = getattr(estimator, "classes_", None)
        if classes is None and hasattr(estimator, "named_steps"):
            for step in reversed(list(estimator.named_steps.values())):
                if hasattr(step, "classes_"):
                    classes = step.classes_
                    break
        if classes is None or list(classes) != [0, 1]:
            raise ValueError("Cannot safely orient decision_function scores to Tumor=1")
        return 1.0 / (1.0 + np.exp(-np.clip(decision, -35, 35)))

    raise ValueError("Estimator has neither predict_proba nor decision_function")


def compute_metrics(y, prob, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_samples": int(len(y)),
        "n_tumor": int(np.sum(y == 1)),
        "n_normal": int(np.sum(y == 0)),
        "ROC_AUC": roc_auc_score(y, prob),
        "PR_AUC_average_precision": average_precision_score(y, prob),
        "average_precision_baseline": float(np.mean(y)),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_tumor": precision_score(y, pred, zero_division=0),
        "sensitivity_tumor": recall_score(y, pred, zero_division=0),
        "specificity_normal": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1_tumor": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "Brier_score": brier_score_loss(y, prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "threshold": THRESHOLD,
    }


# ------------------------------- RUN -------------------------------
print("=" * 78)
print("GSE58135 EXTERNAL VALIDATION — EXISTING TCGA MODELS (READ-ONLY)")
print("=" * 78)
print("Model file:", MODEL_FILE)
print("Feature schema:", SCHEMA_FILE)
print("GEO scores:", SCORE_DIR)

schema = load_schema(SCHEMA_FILE)
metadata = load_metadata(METADATA_FILE)
models = load_models(MODEL_FILE)
print(f"Loaded GEO metadata: {len(metadata)} samples; Tumor={(metadata.y == 1).sum()}, Normal={(metadata.y == 0).sum()}")
print(f"Loaded existing estimators: {len(models)}")

# Load each GEO score matrix once and align to metadata.
score_matrices = {}
for method, filename in SCORE_FILES.items():
    if method == "plage" and not EVALUATE_PLAGE:
        continue
    if method not in schema:
        warnings.warn(f"No TCGA feature schema for {method}; skipping this method")
        continue
    score_df = load_scores(SCORE_DIR / filename, method)
    missing_samples = sorted(set(metadata.index) - set(score_df.index))
    extra_samples = sorted(set(score_df.index) - set(metadata.index))
    if missing_samples or extra_samples:
        raise ValueError(
            f"Sample mismatch for {method}: missing from score matrix={missing_samples[:10]}, "
            f"extra in score matrix={extra_samples[:10]}"
        )
    score_df = score_df.loc[metadata.index]

    expected = schema[method]
    missing_features = [f for f in expected if f not in score_df.columns]
    extra_features = [f for f in score_df.columns if f not in expected]
    if missing_features:
        raise ValueError(f"{method}: missing {len(missing_features)} TCGA features: {missing_features}")
    X_df = score_df.loc[:, expected].copy()  # exact TCGA feature order
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X_df.to_numpy(dtype=float)).all():
        bad = X_df.columns[~np.isfinite(X_df.to_numpy(dtype=float)).all(axis=0)].tolist()
        raise ValueError(f"{method}: non-finite values in aligned features: {bad}")
    score_matrices[method] = (X_df, extra_features)
    print(f"[OK] {method}: {X_df.shape[0]} samples x {X_df.shape[1]} aligned features; ignored extra={len(extra_features)}")

metrics_rows = []
prediction_rows = []
skipped_rows = []

for method, model_name, saved_method, estimator in models:
    if method not in score_matrices:
        skipped_rows.append({"method": saved_method, "model": model_name, "reason": "No aligned GEO pathway matrix/schema"})
        continue
    X_df, extra_features = score_matrices[method]
    y = metadata.loc[X_df.index, "y"].to_numpy(dtype=int)
    X = X_df.to_numpy(dtype=float)

    try:
        prob = positive_probability(estimator, X)
        pred = np.asarray(estimator.predict(X)).astype(int)
        if len(prob) != len(y) or not np.isfinite(prob).all():
            raise ValueError("Invalid probability vector")
        row = {
            "pathway_method": method,
            "saved_pathway_key": saved_method,
            "model": model_name,
            "score_file": SCORE_FILES[method],
            "n_features": X.shape[1],
            "extra_geo_features_ignored": len(extra_features),
            **compute_metrics(y, prob, pred),
        }
        metrics_rows.append(row)
        for sid, actual, p, label in zip(X_df.index, y, prob, pred):
            prediction_rows.append({
                "sample_id": sid,
                "condition": "Tumor" if actual == 1 else "Normal",
                "pathway_method": method,
                "model": model_name,
                "probability_tumor": float(p),
                "predicted_label": "Tumor" if label == 1 else "Normal",
                "threshold": THRESHOLD,
            })

        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{method}_{model_name}")
        fpr, tpr, _ = roc_curve(y, prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC-AUC={row['ROC_AUC']:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False-positive rate")
        plt.ylabel("True-positive rate")
        plt.title(f"GEO ROC — {method} / {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ROC_{tag}.png", dpi=200)
        plt.close()

        precision, recall, _ = precision_recall_curve(y, prob)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AP={row['PR_AUC_average_precision']:.3f}")
        plt.axhline(np.mean(y), linestyle="--", label=f"Prevalence={np.mean(y):.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"GEO PR — {method} / {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"PR_{tag}.png", dpi=200)
        plt.close()

        print(f"[OK] {saved_method}/{model_name}: ROC-AUC={row['ROC_AUC']:.4f}; AP={row['PR_AUC_average_precision']:.4f}")
    except Exception as exc:
        skipped_rows.append({"method": saved_method, "model": model_name, "reason": str(exc)})
        print(f"[SKIP] {saved_method}/{model_name}: {exc}")

metrics_df = pd.DataFrame(metrics_rows)
predictions_df = pd.DataFrame(prediction_rows)
skipped_df = pd.DataFrame(skipped_rows)

metrics_df.to_csv(OUTPUT_DIR / "external_validation_metrics.csv", index=False)
predictions_df.to_csv(OUTPUT_DIR / "external_validation_predictions.csv", index=False)
skipped_df.to_csv(OUTPUT_DIR / "skipped_models.csv", index=False)
if not metrics_df.empty:
    metrics_df.sort_values("PR_AUC_average_precision", ascending=False).to_csv(
        OUTPUT_DIR / "metrics_sorted_by_PR_AUC.csv", index=False
    )

print("\n" + "=" * 78)
print("EXTERNAL VALIDATION FINISHED")
print("Models evaluated:", len(metrics_rows))
print("Models skipped:", len(skipped_rows))
print("Output directory:", OUTPUT_DIR)
if EVALUATE_PLAGE:
    print("WARNING: PLAGE scores are included, but independent-cohort SVD sign orientation remains unverified.")
print("IMPORTANT: Existing TCGA estimators were loaded read-only; no TCGA training or tuning was performed.")
