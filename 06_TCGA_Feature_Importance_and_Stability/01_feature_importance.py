"""
STEP 12 — Feature Importance / SHAP
=====================================
Ranks which pathways drive each model's predictions, using an importance
method matched to the model type:
    - Random Forest, XGBoost  -> shap.TreeExplainer   (exact, fast)
    - Elastic Net             -> standardized coefficients
    - SVM-RBF                 -> permutation importance

Reads directly from your existing artifacts:
    best_estimators.pkl     (nested dict: {pathway: {model_name: Pipeline}})
    nested_cv_splits.pkl    (list of (train_idx, test_idx) tuples)
    FINAL_Pathway_*.csv     (the 5 pathway-scoring outputs)

Produces, per pathway x model combination (20 total):
    <results-dir>/feature_importance/tables/{pathway}_{model}_top_pathways.csv
    <results-dir>/feature_importance/figures/{pathway}_{model}_shap_summary.png   (tree models only)

Run this after 04_TCGA_ML_Benchmarking has produced the .pkl files above.

Usage:
    python 01_feature_importance.py --data-dir data/processed --results-dir results
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance

# ==========================================
# SETTINGS
# ==========================================
parser = argparse.ArgumentParser(description="Feature importance / SHAP for the tuned models.")
parser.add_argument("--data-dir", default="data/processed",
                    help="Folder with FINAL_Pathway_*.csv (default: data/processed).")
parser.add_argument("--results-dir", default="results",
                    help="Folder with models/ and for outputs (default: results).")
args = parser.parse_args()

DATA_DIR = args.data_dir
MODELS_DIR = os.path.join(args.results_dir, "models")
IMPORTANCE_DIR = os.path.join(args.results_dir, "feature_importance", "tables")
FIGURES_DIR = os.path.join(args.results_dir, "feature_importance", "figures")
TOP_N = 20

os.makedirs(IMPORTANCE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Maps sklearn class names (as stored inside your pipelines) to the
# importance method to use for that model type.
MODEL_TYPE_MAP = {
    "RandomForestClassifier": "tree",
    "XGBClassifier": "tree",
    "LogisticRegression": "linear",   # elastic_net is a LogisticRegression instance
    "SVC": "permutation",             # svm_rbf
}

FILE_PATHS = {
    "ssGSEA": os.path.join(DATA_DIR, "FINAL_Pathway_ssgsea.csv"),
    "GSVA": os.path.join(DATA_DIR, "FINAL_Pathway_gsva.csv"),
    "PLAGE": os.path.join(DATA_DIR, "FINAL_Pathway_plage.csv"),
    "ZScore": os.path.join(DATA_DIR, "FINAL_Pathway_zscore.csv"),
    "PROGENy": os.path.join(DATA_DIR, "FINAL_Pathway_PROGENy.csv"),
}


# ==========================================
# DATA LOADING (same loader used in your benchmarking scripts)
# ==========================================
def load_pathway_csv(file_path):
    df = pd.read_csv(file_path)
    df.set_index("File_ID", inplace=True)
    df = df.dropna(subset=["Condition"])
    y = df["Condition"].astype(str).str.strip().map({
        "Tumor": 1, "Solid_Tissue_Normal": 0
    })
    X = df.drop(columns=["Condition"])
    return X, y


def load_and_align_all_pathways(file_paths):
    """Reloads all 5 pathway CSVs WITH column names retained (your original
    benchmarking script converts X to a bare numpy array, which drops the
    pathway names needed for readable importance tables), and re-applies
    the same sample-alignment logic so indices match across methods."""
    X_dict, y_dict = {}, {}
    for method, path in file_paths.items():
        X, y = load_pathway_csv(path)
        X_dict[method] = X
        y_dict[method] = y

    common_samples = sorted(set.intersection(*[set(X.index) for X in X_dict.values()]))
    for method in X_dict:
        X_dict[method] = X_dict[method].loc[common_samples]
        y_dict[method] = y_dict[method].loc[common_samples]

    y = y_dict[list(y_dict.keys())[0]].values
    for method in y_dict:
        assert all(y_dict[method].values == y), f"Label mismatch in {method}"

    return X_dict, y


# ==========================================
# IMPORTANCE FUNCTIONS (matched to model type, aware of Pipeline structure)
# ==========================================
def get_tree_shap_importance(pipeline, X_test_df, top_n=TOP_N, plot_path=None):
    """RF / XGBoost — exact TreeExplainer on the SCALED feature space,
    since that is what the inner model was actually trained on."""
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_scaled = scaler.transform(X_test_df)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # SHAP's return shape for binary classifiers has changed across versions:
    #   - older SHAP: a list [class0_values, class1_values]
    #   - newer SHAP (>=0.45 or so): a single 3D array (n_samples, n_features, n_classes)
    # Handle both so this script works regardless of installed SHAP version.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]           # SHAP values for the "Tumor" (positive) class
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]      # same idea, indexed on the last axis

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "pathway": X_test_df.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    if plot_path:
        shap.summary_plot(shap_values, X_scaled, feature_names=X_test_df.columns, show=False)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    return importance_df.head(top_n)


def get_elasticnet_importance(pipeline, feature_names, top_n=TOP_N):
    """Elastic Net — standardized coefficients from the inner LogisticRegression.
    Assumes features were standardized before fitting (they are, via the
    pipeline's own StandardScaler step), so coefficient magnitude is directly
    comparable across pathways."""
    model = pipeline.named_steps["model"]
    coefs = model.coef_.ravel()
    importance_df = pd.DataFrame({
        "pathway": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)
    return importance_df.head(top_n)


def get_svm_permutation_importance(pipeline, X_test_df, y_test, top_n=TOP_N,
                                    n_repeats=30, random_state=42):
    """SVM-RBF — permutation importance on the FULL pipeline with RAW features.
    Do not pre-scale X: the pipeline's own scaler step handles that internally
    on each call; pre-scaling would double-transform the data."""
    result = permutation_importance(
        pipeline, X_test_df, y_test,
        scoring="average_precision",   # matches PR-AUC, appropriate given class imbalance
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "pathway": X_test_df.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    return importance_df.head(top_n)


# ==========================================
# MAIN
# ==========================================
def main():
    print("[INFO] Loading pathway CSVs with feature names retained...")
    X_dict, y = load_and_align_all_pathways(FILE_PATHS)

    print("[INFO] Loading best_estimators.pkl and nested_cv_splits.pkl...")
    with open(os.path.join(MODELS_DIR, "best_estimators.pkl"), "rb") as f:
        best_estimators = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "nested_cv_splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    # Fold 0 is used as a single, fixed, known train/test split for every
    # pathway/model combination. This resolves a gap in best_estimators.pkl:
    # it stores a FITTED pipeline but not which fold produced that fit, so we
    # clone (keep hyperparameters, discard the fit) and refit on a split we
    # control -- the same split every time, so results are fairly comparable
    # across all 20 pathway x model combinations.
    train_idx, test_idx = splits[0]
    print(f"[INFO] Using fold 0 as evaluation split: "
          f"{len(train_idx)} train / {len(test_idx)} test samples")

    importance_results = {}

    for pathway_name, models in best_estimators.items():
        X_test_df = X_dict[pathway_name].iloc[test_idx]
        X_train_df = X_dict[pathway_name].iloc[train_idx]
        y_test = y[test_idx]
        y_train = y[train_idx]

        for model_name, stored_pipeline in models.items():
            model_class = type(stored_pipeline.named_steps["model"]).__name__
            kind = MODEL_TYPE_MAP.get(model_class)

            if kind is None:
                print(f"[WARN] No importance method mapped for {model_class} "
                      f"({pathway_name}/{model_name}) -- skipping")
                continue

            tag = f"{pathway_name}_{model_name}"
            print(f"\n[RUNNING] {tag}  (model type: {model_class} -> {kind})")

            # Refit a clone on the designated split -- see comment above.
            pipeline = clone(stored_pipeline)
            pipeline.fit(X_train_df, y_train)

            if kind == "tree":
                imp_df = get_tree_shap_importance(
                    pipeline, X_test_df,
                    plot_path=f"{FIGURES_DIR}/{tag}_shap_summary.png"
                )
            elif kind == "linear":
                imp_df = get_elasticnet_importance(pipeline, X_test_df.columns)
            else:  # permutation
                imp_df = get_svm_permutation_importance(pipeline, X_test_df, y_test)

            imp_df.to_csv(f"{IMPORTANCE_DIR}/{tag}_top_pathways.csv", index=False)
            importance_results[tag] = imp_df

            print(f"Top 10 pathways -- {tag}:")
            print(imp_df.head(10).to_string(index=False))

    print(f"\n[DONE] Wrote {len(importance_results)} importance tables to "
          f"{IMPORTANCE_DIR}/, and SHAP plots to {FIGURES_DIR}/")

    return importance_results


if __name__ == "__main__":
    main()
