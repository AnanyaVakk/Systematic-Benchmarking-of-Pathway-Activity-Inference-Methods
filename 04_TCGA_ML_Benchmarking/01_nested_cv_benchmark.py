"""
04 - Nested cross-validation benchmark of pathway-activity features.

Compares 5 pathway scoring methods (ssGSEA, GSVA, PLAGE, Z-score, PROGENy) with
4 classifiers (elastic net, random forest, RBF-SVM, XGBoost) for Tumor vs
Solid_Tissue_Normal, using repeated stratified outer CV and randomized-search
inner CV (PR-AUC).

Inputs  (in --data-dir):   FINAL_Pathway_{ssgsea,gsva,plage,zscore,PROGENy}.csv
Outputs (in --results-dir):
    models/nested_cv_splits.pkl     (reused by steps 06)
    models/best_estimators.pkl      (reused by steps 05, 06)
    benchmarking/benchmark_summary.csv

Usage:
    python 01_nested_cv_benchmark.py --data-dir data/processed --results-dir results
"""
import argparse
import os
# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import pandas as pd
import pickle

from scipy.stats import loguniform, randint, uniform

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss
)

# ==========================================
# SETTINGS
# ==========================================
parser = argparse.ArgumentParser(
    description="Nested-CV benchmark: 5 pathway-scoring methods x 4 classifiers.")
parser.add_argument("--data-dir", default="data/processed",
                    help="Folder with FINAL_Pathway_*.csv (default: data/processed).")
parser.add_argument("--results-dir", default="results",
                    help="Folder for outputs (default: results).")
args = parser.parse_args()

DATA_DIR = args.data_dir
MODELS_DIR = os.path.join(args.results_dir, "models")
BENCH_DIR = os.path.join(args.results_dir, "benchmarking")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)
SPLITS_PATH = os.path.join(MODELS_DIR, "nested_cv_splits.pkl")
BEST_ESTIMATORS_PATH = os.path.join(MODELS_DIR, "best_estimators.pkl")

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_REPEATS = 3
N_INNER_SPLITS = 3
N_ITER_SEARCH = 30
VERBOSE = True

def vprint(msg):
    if VERBOSE:
        print(msg)

# ==========================================
# DATA LOADING
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
# PROVIDE FILE PATHS HERE
# ==========================================
file_paths = {
    "ssGSEA": os.path.join(DATA_DIR, "FINAL_Pathway_ssgsea.csv"),
    "GSVA": os.path.join(DATA_DIR, "FINAL_Pathway_gsva.csv"),
    "PLAGE": os.path.join(DATA_DIR, "FINAL_Pathway_plage.csv"),
    "ZScore": os.path.join(DATA_DIR, "FINAL_Pathway_zscore.csv"),
    "PROGENy": os.path.join(DATA_DIR, "FINAL_Pathway_PROGENy.csv")
}

# ==========================================
# LOAD ALL METHODS
# ==========================================
X_dict = {}
y_dict = {}

for method, path in file_paths.items():
    print(f"[INFO] Loading {method} from {path}")
    X, y = load_pathway_csv(path)

    print(f"[INFO] {method} shape: {X.shape}")

    X_dict[method] = X
    y_dict[method] = y


# ==========================================
# ALIGN SAMPLES ACROSS ALL METHODS
# ==========================================
print("\n[INFO] Aligning samples across methods...")

common_samples = None

for method, X in X_dict.items():
    if common_samples is None:
        common_samples = set(X.index)
    else:
        common_samples = common_samples.intersection(set(X.index))

print(f"[INFO] Common samples found: {len(common_samples)}")

# Sort for consistency
common_samples = sorted(list(common_samples))

# Subset all datasets
for method in X_dict:
    X_dict[method] = X_dict[method].loc[common_samples]
    y_dict[method] = y_dict[method].loc[common_samples]

# ==========================================
# FINAL LABEL VECTOR (SINGLE SOURCE OF TRUTH)
# ==========================================
y = y_dict[list(y_dict.keys())[0]].values

print(f"[INFO] Final dataset size: {len(y)} samples")

# Optional sanity check
for method in y_dict:
    assert all(y_dict[method].values == y), f"Label mismatch in {method}"

# ==========================================
# CV SETUP
# ==========================================
outer_cv = RepeatedStratifiedKFold(
    n_splits=N_OUTER_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)

inner_cv = StratifiedKFold(
    n_splits=N_INNER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# ==========================================
# GENERATE & SAVE SPLITS
# ==========================================
splits = list(outer_cv.split(list(X_dict.values())[0], y))

with open(SPLITS_PATH, "wb") as f:
    pickle.dump(splits, f)

vprint("Splits saved!")

# ==========================================
# LOAD SPLITS
# ==========================================
with open(SPLITS_PATH, "rb") as f:
    splits = pickle.load(f)

vprint(f"Loaded {len(splits)} splits")

# ==========================================
# MODELS
# ==========================================
def get_models(random_state):

    return {

        "elastic_net": (
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state
            ),
            {
                "model__C": loguniform(1e-3, 1e2),
                "model__l1_ratio": uniform(0.05, 0.9)
            }
        ),

        "rf": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1
            ),
            {
                "model__n_estimators": randint(200, 800),
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": randint(2, 10),
                "model__min_samples_leaf": randint(1, 5)
            }
        ),

        "svm_rbf": (
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state
            ),
            {
                "model__C": loguniform(1e-3, 1e2),
                "model__gamma": loguniform(1e-4, 1e0)
            }
        ),

        "xgb": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1
            ),
            {
                "model__max_depth": randint(3, 8),
                "model__learning_rate": loguniform(1e-3, 0.2),
                "model__n_estimators": randint(200, 800),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4)
            }
        )
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

        # Standard
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),

        # Macro (IMPORTANT for imbalance)
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "Precision_macro": precision_score(y_true, y_pred, average="macro"),
        "Recall_macro": recall_score(y_true, y_pred, average="macro"),

        # Weighted
        "F1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "Precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "Recall_weighted": recall_score(y_true, y_pred, average="weighted"),

        # Others
        "Specificity": specificity,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob)
    }

# ==========================================
# MAIN BENCHMARK LOOP
# ==========================================
results = []
models = get_models(RANDOM_STATE)

best_estimators = {
    pathway: {model_name: None for model_name in get_models(RANDOM_STATE)}
    for pathway in X_dict
}

best_scores = {
    pathway: {model_name: -np.inf for model_name in get_models(RANDOM_STATE)}
    for pathway in X_dict
}

for pathway_name, X in X_dict.items():

    vprint(f"\n===== {pathway_name} =====")
    X = np.array(X)

    for fold_id, (train_idx, test_idx) in enumerate(splits):

        vprint(f"\n--- Fold {fold_id+1} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fold-specific imbalance weight
        pos = np.sum(y_train)
        neg = len(y_train) - pos
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        for model_name, (base_model, param_dist) in models.items():

            vprint(f"Training {model_name}")

            model = base_model

            if model_name == "xgb":
                model.set_params(scale_pos_weight=scale_pos_weight)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=N_ITER_SEARCH,
                scoring="average_precision",
                cv=inner_cv,
                n_jobs=-1,
                random_state=RANDOM_STATE
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_score = search.best_score_

            # Update global best per pathway + model
            if best_score > best_scores[pathway_name][model_name]:
                best_scores[pathway_name][model_name] = best_score
                best_estimators[pathway_name][model_name] = best_model

            y_prob = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)

            metrics = compute_metrics(y_test, y_prob, y_pred)

            row = {
                "Pathway": pathway_name,
                "Model": model_name,
                "Fold": fold_id,
                "Best_Params": search.best_params_,
                "InnerCV_PR_AUC": search.best_score_
            }

            row.update(metrics)
            results.append(row)

# ==========================================
# SAVE RESULTS
# ==========================================
with open(BEST_ESTIMATORS_PATH, "wb") as f:
    pickle.dump(best_estimators, f)

print(f"[INFO] Best estimators saved to {BEST_ESTIMATORS_PATH}")

results_df = pd.DataFrame(results)

# Select only numeric columns
numeric_cols = results_df.select_dtypes(include=np.number).columns

# Aggregate safely
summary_df = (
    results_df
    .groupby(["Pathway", "Model"])[numeric_cols]
    .agg(["mean", "std"])
)

summary_df.to_csv(os.path.join(BENCH_DIR, "benchmark_summary.csv"))

print("\n===== DONE =====")
print(summary_df)