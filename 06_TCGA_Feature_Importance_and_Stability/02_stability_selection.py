"""
STEP 13 — Stability Selection Analysis (corrected)
===================================================

Corrections implemented
-----------------------
1. Nogueira stability uses the FULL pathway feature space, including
   pathways that were never selected.
2. SVM-RBF permutation importance is computed OUT-OF-SAMPLE using an
   internal validation split of each stability subsample.
3. Stability is evaluated across ALL outer CV training folds instead of
   using fold 0 only. The held-out test portion of each fold is never used.
4. The iteration count is convergence-controlled:
      - MAX_ITERATIONS_PER_FOLD is the upper limit (100 by default)
      - metrics are checked at checkpoints
      - the loop stops early when selection frequencies, Jaccard, and
        Nogueira stability have converged.
   Therefore, 100 is a maximum, not a mandatory number of iterations.

Saved outputs are intentionally compact:
    <results-dir>/stability/stability_summary_all_methods.csv
        -> one row per pathway-method/model combination

    <results-dir>/stability/stability_pathway_details.csv
        -> only the top pathways by selection frequency for each combination,
           with an indicator for >=80% selection stability

No separate CSV is created for every model/pathway combination.
"""

import argparse
import os
import pickle
import time
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import shap

from scipy.stats import entropy
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# ============================================================
# SETTINGS
# ============================================================

parser = argparse.ArgumentParser(description="Stability selection of pathway importance.")
parser.add_argument("--data-dir", default="data/processed",
                    help="Folder with FINAL_Pathway_*.csv (default: data/processed).")
parser.add_argument("--results-dir", default="results",
                    help="Folder with models/ and for outputs (default: results).")
args = parser.parse_args()

DATA_DIR = args.data_dir
MODELS_DIR = os.path.join(args.results_dir, "models")
STABILITY_DIR = os.path.join(args.results_dir, "stability")
os.makedirs(STABILITY_DIR, exist_ok=True)

# ------------------------------------------------------------
# Stability iterations
# ------------------------------------------------------------
# 100 is now a MAXIMUM, not a compulsory number.
# The code stops early when the stability estimates converge.
MAX_ITERATIONS_PER_FOLD = 80

# Do not test convergence before this many iterations.
MIN_ITERATIONS_PER_FOLD = 20

# Check convergence every N iterations.
CONVERGENCE_CHECK_EVERY = 10

# Require this many consecutive converged checkpoints before stopping.
CONVERGENCE_PATIENCE = 2

# Tolerances for convergence.
# These are changes between two checkpoints, not absolute targets.
FREQ_CONVERGENCE_TOL = 0.02
JACCARD_CONVERGENCE_TOL = 0.01
NOGUEIRA_CONVERGENCE_TOL = 0.02

# ------------------------------------------------------------
# Stability selection
# ------------------------------------------------------------
TOP_K = 20
SUBSAMPLE_FRAC = 0.80
STABLE_FREQ_THRESHOLD = 0.80

# ------------------------------------------------------------
# SVM permutation importance
# ------------------------------------------------------------
# Importance is calculated on a held-out 20% internal validation split
# of every stability subsample.
SVM_IMPORTANCE_VAL_FRAC = 0.20
SVM_PERMUTATION_REPEATS = 5

# ------------------------------------------------------------
# Outer folds
# ------------------------------------------------------------
# None = use ALL outer folds from nested_cv_splits.pkl.
# For a quick sanity check you can temporarily use:
# STABILITY_OUTER_FOLDS = [0, 1]
STABILITY_OUTER_FOLDS = None

RANDOM_STATE = 42


MODEL_TYPE_MAP = {
    "RandomForestClassifier": "tree",
    "XGBClassifier": "tree",
    "LogisticRegression": "linear",
    "SVC": "permutation",
}


FILE_PATHS = {
    "ssGSEA": os.path.join(DATA_DIR, "FINAL_Pathway_ssgsea.csv"),
    "GSVA": os.path.join(DATA_DIR, "FINAL_Pathway_gsva.csv"),
    "PLAGE": os.path.join(DATA_DIR, "FINAL_Pathway_plage.csv"),
    "ZScore": os.path.join(DATA_DIR, "FINAL_Pathway_zscore.csv"),
    "PROGENy": os.path.join(DATA_DIR, "FINAL_Pathway_PROGENy.csv"),
}


# ============================================================
# DATA LOADING
# ============================================================

def load_pathway_csv(file_path):
    df = pd.read_csv(file_path)

    if "File_ID" not in df.columns:
        raise ValueError(f"{file_path} is missing the 'File_ID' column.")

    if "Condition" not in df.columns:
        raise ValueError(f"{file_path} is missing the 'Condition' column.")

    df.set_index("File_ID", inplace=True)
    df = df.dropna(subset=["Condition"])

    y = (
        df["Condition"]
        .astype(str)
        .str.strip()
        .map({
            "Tumor": 1,
            "Solid_Tissue_Normal": 0,
        })
    )

    if y.isna().any():
        bad_labels = sorted(
            df.loc[y.isna(), "Condition"].astype(str).unique().tolist()
        )
        raise ValueError(
            f"Unrecognized Condition labels in {file_path}: {bad_labels}"
        )

    X = df.drop(columns=["Condition"])

    # Ensure all pathway values are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError(
            f"NaN/non-numeric values detected in pathway matrix: {file_path}"
        )

    return X, y.astype(int)


def load_and_align_all_pathways(file_paths):
    X_dict = {}
    y_dict = {}

    for method, path in file_paths.items():
        print(f"[INFO] Loading {method}: {path}")
        X, y = load_pathway_csv(path)
        X_dict[method] = X
        y_dict[method] = y

    common_samples = sorted(
        set.intersection(*[set(X.index) for X in X_dict.values()])
    )

    if len(common_samples) == 0:
        raise ValueError("No common samples found across pathway matrices.")

    for method in X_dict:
        X_dict[method] = X_dict[method].loc[common_samples]
        y_dict[method] = y_dict[method].loc[common_samples]

    y_reference = y_dict[list(y_dict.keys())[0]].values

    for method in y_dict:
        if not np.array_equal(y_dict[method].values, y_reference):
            raise ValueError(f"Label mismatch detected in {method}")

    print(f"[INFO] Common samples across all pathway methods: {len(common_samples)}")

    return X_dict, y_reference


# ============================================================
# TOP-K SIZING
# ============================================================

def effective_top_k(n_features, requested_k=TOP_K, max_frac=0.50, min_k=3):
    """
    Keep top-K as a genuine subset of the feature space.

    This is especially important for PROGENy, which has far fewer
    pathways than the Hallmark-based methods.
    """
    if n_features <= 0:
        raise ValueError("Number of features must be > 0.")

    if n_features < min_k:
        return n_features

    cap = max(min_k, int(n_features * max_frac))
    return min(requested_k, cap)

# ============================================================
# IMPORTANCE METHODS
# ============================================================

def get_top_k_features(
    pipeline,
    X_eval,
    y_eval,
    kind,
    k,
    permutation_random_state=42,
):
    """
    Extract the top-K pathway features.

    tree:
        mean absolute SHAP value

    linear:
        absolute model coefficients

    permutation:
        permutation importance on X_eval/y_eval

    IMPORTANT:
        For SVM, X_eval/y_eval are an INTERNAL HELD-OUT VALIDATION SET,
        not the same data used to fit the pipeline.
    """

    if kind == "tree":
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["model"]

        X_scaled = scaler.transform(X_eval)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        # SHAP compatibility across versions.
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]

        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Binary classifier: class 1.
            shap_values = shap_values[:, :, 1]

        scores = np.abs(np.asarray(shap_values)).mean(axis=0)

    elif kind == "linear":
        model = pipeline.named_steps["model"]
        scores = np.abs(model.coef_.ravel())

    elif kind == "permutation":
        result = permutation_importance(
            pipeline,
            X_eval,
            y_eval,
            scoring="average_precision",
            n_repeats=SVM_PERMUTATION_REPEATS,
            random_state=permutation_random_state,
            n_jobs=-1,
        )
        scores = result.importances_mean

    else:
        raise ValueError(f"Unknown importance kind: {kind}")

    if len(scores) != X_eval.shape[1]:
        raise ValueError(
            f"Importance length ({len(scores)}) does not match "
            f"number of features ({X_eval.shape[1]})."
        )

    ranked = (
        pd.Series(scores, index=X_eval.columns)
        .sort_values(ascending=False)
    )

    return set(ranked.head(k).index)

# ============================================================
# STABILITY METRICS
# ============================================================

def nogueira_stability(topk_sets, all_features):
    """
    Nogueira, Sechidis & Brown-style chance-corrected stability.

    IMPORTANT FIX:
        all_features MUST be the complete pathway feature space,
        not only the union of features selected at least once.

    This means pathways that were NEVER selected still contribute
    zero-selection indicators to the stability calculation.
    """

    if len(topk_sets) < 2:
        return np.nan

    all_features = list(all_features)

    if len(all_features) == 0:
        return np.nan

    Z = np.array(
        [
            [1 if feature in selected else 0 for feature in all_features]
            for selected in topk_sets
        ],
        dtype=float,
    )

    k_bar = Z.sum(axis=1).mean()
    d = len(all_features)

    if d == 0:
        return np.nan

    selection_fraction = k_bar / d
    denominator = selection_fraction * (1.0 - selection_fraction)

    if denominator <= 0:
        return np.nan

    mean_variance = Z.var(axis=0, ddof=1).mean()

    return float(1.0 - (mean_variance / denominator))


def calculate_stability_metrics(topk_sets, all_features):
    """
    Calculate all stability metrics from the current collection
    of top-K selections.
    """

    all_features = list(all_features)
    n_runs = len(topk_sets)

    if n_runs == 0:
        raise ValueError("No top-K selections available.")

    selection_counter = Counter()
    selection_counter.update(
        feature
        for selected_set in topk_sets
        for feature in selected_set
    )

    selection_freq = (
        pd.Series(
            {
                feature: selection_counter.get(feature, 0) / n_runs
                for feature in all_features
            },
            dtype=float,
        )
        .sort_values(ascending=False)
    )

    # Average pairwise Jaccard overlap.
    if n_runs >= 2:
        jaccard_scores = [
            len(a & b) / len(a | b)
            for a, b in combinations(topk_sets, 2)
            if len(a | b) > 0
        ]
        avg_jaccard = float(np.mean(jaccard_scores))
    else:
        avg_jaccard = np.nan

    nogueira = nogueira_stability(
        topk_sets,
        all_features=all_features,
    )

    # Shannon entropy of the selection-frequency distribution.
    # Only features selected at least once enter the distribution.
    freq_nonzero = selection_freq[selection_freq > 0]

    if len(freq_nonzero) > 0:
        normalized_freq = freq_nonzero / freq_nonzero.sum()
        selection_entropy = float(entropy(normalized_freq))
    else:
        selection_entropy = np.nan

    return {
        "selection_freq": selection_freq,
        "avg_jaccard": avg_jaccard,
        "nogueira_stability": nogueira,
        "selection_entropy": selection_entropy,
    }


def convergence_reached(
    current_metrics,
    previous_metrics,
):
    """
    Decide whether two consecutive checkpoints are sufficiently close.

    All three criteria must be within their respective tolerances.
    """

    if previous_metrics is None:
        return False, None

    current_freq = current_metrics["selection_freq"]
    previous_freq = previous_metrics["selection_freq"]

    freq_delta = float(
        (current_freq - previous_freq)
        .abs()
        .max()
    )

    jaccard_delta = abs(
        current_metrics["avg_jaccard"]
        - previous_metrics["avg_jaccard"]
    )

    nogueira_delta = abs(
        current_metrics["nogueira_stability"]
        - previous_metrics["nogueira_stability"]
    )

    converged = (
        freq_delta <= FREQ_CONVERGENCE_TOL
        and jaccard_delta <= JACCARD_CONVERGENCE_TOL
        and nogueira_delta <= NOGUEIRA_CONVERGENCE_TOL
    )

    diagnostics = {
        "max_selection_frequency_delta": freq_delta,
        "jaccard_delta": float(jaccard_delta),
        "nogueira_delta": float(nogueira_delta),
        "converged": bool(converged),
    }

    return bool(converged), diagnostics


# ============================================================
# ONE STABILITY ANALYSIS
# ============================================================

def stability_analysis(
    X_train_df,
    y_train,
    stored_pipeline,
    kind,
    pathway_name,
    model_name,
    n_iterations=MAX_ITERATIONS_PER_FOLD,
    top_k=TOP_K,
    subsample_frac=SUBSAMPLE_FRAC,
    random_state=RANDOM_STATE,
):
    """
    Run stability selection on ONE outer training fold.

    For each iteration:
        1. draw an 80% stratified subsample
        2. refit a fresh clone
        3. obtain top-K pathways
        4. store the selection set

    SVM special handling:
        - the 80% stability subsample is split again into
          80% internal-fit / 20% internal-validation
        - SVM is fitted on internal-fit
        - permutation importance is evaluated on internal-validation

    The outer test fold is NEVER passed here.
    """

    rng = np.random.RandomState(random_state)

    topk_sets = []

    k_used = effective_top_k(
        X_train_df.shape[1],
        requested_k=top_k,
    )

    if k_used != top_k:
        print(
            f"    [INFO] {pathway_name}: {X_train_df.shape[1]} pathways; "
            f"using top_k={k_used} instead of {top_k}"
        )

    previous_checkpoint_metrics = None
    consecutive_converged = 0
    actual_iterations = 0

    final_metrics = None
    convergence_diagnostics = None

    for i in range(n_iterations):

        iteration_seed = int(rng.randint(1_000_000_000))

        # ----------------------------------------------------
        # 80% stratified subsample WITHOUT replacement.
        # This is subsampling, not bootstrap sampling.
        # ----------------------------------------------------
        X_sub, y_sub = resample(
            X_train_df,
            y_train,
            replace=False,
            n_samples=max(
                2,
                int(subsample_frac * len(X_train_df)),
            ),
            stratify=y_train,
            random_state=iteration_seed,
        )

        pipeline_i = clone(stored_pipeline)

        # ----------------------------------------------------
        # FIX #2:
        # SVM permutation importance must be evaluated on data
        # not used for fitting.
        # ----------------------------------------------------
        if kind == "permutation":

            X_fit, X_eval, y_fit, y_eval = train_test_split(
                X_sub,
                y_sub,
                test_size=SVM_IMPORTANCE_VAL_FRAC,
                stratify=y_sub,
                random_state=iteration_seed + 1,
            )

            pipeline_i.fit(X_fit, y_fit)

            top_k_set = get_top_k_features(
                pipeline_i,
                X_eval=X_eval,
                y_eval=y_eval,
                kind=kind,
                k=k_used,
                permutation_random_state=iteration_seed + 2,
            )

        else:
            # Tree SHAP and linear coefficients are model-internal
            # importance measures, so fit on the complete stability
            # subsample.
            pipeline_i.fit(X_sub, y_sub)

            top_k_set = get_top_k_features(
                pipeline_i,
                X_eval=X_sub,
                y_eval=y_sub,
                kind=kind,
                k=k_used,
                permutation_random_state=iteration_seed + 2,
            )

        topk_sets.append(top_k_set)
        actual_iterations += 1

        # ----------------------------------------------------
        # FIX #4:
        # Check convergence instead of blindly requiring 100.
        # ----------------------------------------------------
        if (
            actual_iterations >= MIN_ITERATIONS_PER_FOLD
            and actual_iterations % CONVERGENCE_CHECK_EVERY == 0
        ):
            current_metrics = calculate_stability_metrics(
                topk_sets,
                all_features=X_train_df.columns,
            )

            converged, diagnostics = convergence_reached(
                current_metrics,
                previous_checkpoint_metrics,
            )

            convergence_diagnostics = diagnostics

            if diagnostics is None:
                print(
                    f"    checkpoint={actual_iterations} | "
                    f"Jaccard={current_metrics['avg_jaccard']:.4f} | "
                    f"Nogueira={current_metrics['nogueira_stability']:.4f} | "
                    f"baseline checkpoint"
                )
            else:
                print(
                    f"    checkpoint={actual_iterations} | "
                    f"Jaccard={current_metrics['avg_jaccard']:.4f} | "
                    f"Nogueira={current_metrics['nogueira_stability']:.4f} | "
                    f"max_freq_delta="
                    f"{diagnostics['max_selection_frequency_delta']:.4f}"
                )

            if converged:
                consecutive_converged += 1
            else:
                consecutive_converged = 0

            previous_checkpoint_metrics = current_metrics

            if consecutive_converged >= CONVERGENCE_PATIENCE:
                print(
                    f"    [INFO] Converged after {actual_iterations} "
                    f"iterations; stopping before the maximum of "
                    f"{n_iterations}."
                )
                break

    final_metrics = calculate_stability_metrics(
        topk_sets,
        all_features=X_train_df.columns,
    )

    return {
        "topk_sets": topk_sets,
        "selection_freq": final_metrics["selection_freq"],
        "avg_jaccard": final_metrics["avg_jaccard"],
        "nogueira_stability": final_metrics["nogueira_stability"],
        "selection_entropy": final_metrics["selection_entropy"],
        "top_k_used": k_used,
        "n_pathways_total": X_train_df.shape[1],
        "n_iterations_actual": actual_iterations,
        "convergence_diagnostics": convergence_diagnostics,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STEP 13 — CORRECTED STABILITY SELECTION")
    print("=" * 80)

    print("\n[INFO] Loading pathway matrices...")
    X_dict, y = load_and_align_all_pathways(FILE_PATHS)

    print("\n[INFO] Loading best_estimators.pkl...")
    with open(os.path.join(MODELS_DIR, "best_estimators.pkl"), "rb") as f:
        best_estimators = pickle.load(f)

    print("[INFO] Loading nested_cv_splits.pkl...")
    with open(os.path.join(MODELS_DIR, "nested_cv_splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    # --------------------------------------------------------
    # FIX #3:
    # Use ALL outer folds instead of fold 0 only.
    # --------------------------------------------------------
    if STABILITY_OUTER_FOLDS is None:
        outer_fold_indices = list(range(len(splits)))
    else:
        outer_fold_indices = list(STABILITY_OUTER_FOLDS)

    if len(outer_fold_indices) == 0:
        raise ValueError("No outer folds selected for stability analysis.")

    for fold_idx in outer_fold_indices:
        if fold_idx < 0 or fold_idx >= len(splits):
            raise IndexError(
                f"Invalid outer fold index {fold_idx}; "
                f"available folds: 0..{len(splits)-1}"
            )

    print(
        f"\n[INFO] Using outer folds: {outer_fold_indices} "
        f"({len(outer_fold_indices)} folds)"
    )

    print(
        f"[INFO] Maximum iterations per fold: "
        f"{MAX_ITERATIONS_PER_FOLD}"
    )
    print(
        f"[INFO] Minimum iterations before convergence checks: "
        f"{MIN_ITERATIONS_PER_FOLD}"
    )
    print(
        f"[INFO] Subsample fraction: {SUBSAMPLE_FRAC:.0%}"
    )

    # One summary row per pathway method x model.
    stability_results = []

    # One compact pathway-level table containing only the most important
    # pathways from each combination.
    pathway_detail_rows = []

    n_combos = sum(len(models) for models in best_estimators.values())
    combo_num = 0

    # --------------------------------------------------------
    # Loop over pathway methods and models.
    # --------------------------------------------------------
    for pathway_name, models in best_estimators.items():

        if pathway_name not in X_dict:
            print(
                f"[WARN] {pathway_name} is present in best_estimators.pkl "
                f"but not in FILE_PATHS. Skipping."
            )
            continue

        for model_name, stored_pipeline in models.items():

            model_class = type(
                stored_pipeline.named_steps["model"]
            ).__name__

            kind = MODEL_TYPE_MAP.get(model_class)

            if kind is None:
                print(
                    f"[WARN] No importance method mapped for "
                    f"{model_class} "
                    f"({pathway_name}/{model_name}) -- skipping"
                )
                continue

            combo_num += 1

            print("\n" + "=" * 80)
            print(
                f"[{combo_num}/{n_combos}] "
                f"{pathway_name} + {model_name}"
            )
            print("=" * 80)

            combo_start = time.time()

            # ------------------------------------------------
            # Accumulate top-K selections across ALL outer folds.
            # ------------------------------------------------
            combined_topk_sets = []
            combined_selection_counter = Counter()

            fold_iteration_counts = []
            fold_jaccards = []
            fold_nogueiras = []

            n_pathways_total = X_dict[pathway_name].shape[1]
            k_used = effective_top_k(
                n_pathways_total,
                requested_k=TOP_K,
            )

            # -----------------------------------------------
            # Each outer fold supplies its TRAINING portion.
            # Its TEST portion is never touched.
            # -----------------------------------------------
            for fold_idx in outer_fold_indices:

                train_idx, test_idx = splits[fold_idx]

                X_train_df = X_dict[pathway_name].iloc[train_idx]
                y_train = y[train_idx]

                print(
                    f"\n  [Fold {fold_idx + 1}/{len(splits)}] "
                    f"training samples={len(train_idx)} | "
                    f"test samples={len(test_idx)}"
                )

                fold_start = time.time()

                fold_result = stability_analysis(
                    X_train_df=X_train_df,
                    y_train=y_train,
                    stored_pipeline=stored_pipeline,
                    kind=kind,
                    pathway_name=pathway_name,
                    model_name=model_name,
                    n_iterations=MAX_ITERATIONS_PER_FOLD,
                    top_k=TOP_K,
                    subsample_frac=SUBSAMPLE_FRAC,
                    random_state=RANDOM_STATE + (fold_idx * 10000),
                )

                combined_topk_sets.extend(
                    fold_result["topk_sets"]
                )

                combined_selection_counter.update(
                    feature
                    for selected_set in fold_result["topk_sets"]
                    for feature in selected_set
                )

                fold_iteration_counts.append(
                    fold_result["n_iterations_actual"]
                )

                fold_jaccards.append(
                    fold_result["avg_jaccard"]
                )

                fold_nogueiras.append(
                    fold_result["nogueira_stability"]
                )

                print(
                    f"    Fold result: "
                    f"iterations={fold_result['n_iterations_actual']} | "
                    f"Jaccard={fold_result['avg_jaccard']:.4f} | "
                    f"Nogueira={fold_result['nogueira_stability']:.4f} | "
                    f"time={time.time() - fold_start:.1f}s"
                )

            # ------------------------------------------------
            # Aggregate across ALL selected outer folds.
            # ------------------------------------------------
            n_total_runs = len(combined_topk_sets)

            if n_total_runs == 0:
                print(
                    f"[WARN] No stability runs completed for "
                    f"{pathway_name}/{model_name}"
                )
                continue

            combined_metrics = calculate_stability_metrics(
                combined_topk_sets,
                all_features=X_dict[pathway_name].columns,
            )

            selection_freq = combined_metrics["selection_freq"]

            # ------------------------------------------------
            # Stable pathways = >=80% of all resampling runs.
            # ------------------------------------------------
            stable_pathways = selection_freq[
                selection_freq >= STABLE_FREQ_THRESHOLD
            ]

            # ------------------------------------------------
            # Save ONLY important pathway details:
            # top TOP_K by selection frequency.
            #
            # We do NOT save a huge full feature-by-feature file.
            # ------------------------------------------------
            top_pathways = selection_freq.head(
                min(TOP_K, len(selection_freq))
            )

            for rank, (pathway, freq) in enumerate(
                top_pathways.items(),
                start=1,
            ):
                pathway_detail_rows.append(
                    {
                        "pathway_method": pathway_name,
                        "model": model_name,
                        "pathway": pathway,
                        "rank_by_selection_frequency": rank,
                        "selection_frequency": float(freq),
                        "selected_in_runs": int(round(freq * n_total_runs)),
                        "n_total_runs": n_total_runs,
                        "stable_ge80pct": bool(
                            freq >= STABLE_FREQ_THRESHOLD
                        ),
                    }
                )

            # ------------------------------------------------
            # Headline summary.
            # ------------------------------------------------
            combo_time = time.time() - combo_start

            stability_results.append(
                {
                    "pathway_method": pathway_name,
                    "model": model_name,
                    "model_class": model_class,
                    "top_k_used": k_used,
                    "n_pathways_total": n_pathways_total,
                    "subsample_fraction": SUBSAMPLE_FRAC,
                    "stable_threshold": STABLE_FREQ_THRESHOLD,
                    "n_outer_folds_used": len(outer_fold_indices),
                    "outer_folds_used": ",".join(
                        str(x) for x in outer_fold_indices
                    ),
                    "n_total_runs": n_total_runs,
                    "min_iterations_per_fold": MIN_ITERATIONS_PER_FOLD,
                    "max_iterations_per_fold": MAX_ITERATIONS_PER_FOLD,
                    "mean_iterations_per_fold": float(
                        np.mean(fold_iteration_counts)
                    ),
                    "avg_jaccard": combined_metrics["avg_jaccard"],
                    "nogueira_stability": combined_metrics[
                        "nogueira_stability"
                    ],
                    "selection_entropy": combined_metrics[
                        "selection_entropy"
                    ],
                    "n_stable_pathways_ge80pct": int(
                        len(stable_pathways)
                    ),
                    "best_pathway": (
                        selection_freq.index[0]
                        if len(selection_freq) > 0
                        else None
                    ),
                    "best_pathway_selection_frequency": (
                        float(selection_freq.iloc[0])
                        if len(selection_freq) > 0
                        else np.nan
                    ),
                    "mean_fold_jaccard": float(
                        np.mean(fold_jaccards)
                    ),
                    "mean_fold_nogueira": float(
                        np.mean(fold_nogueiras)
                    ),
                    "runtime_seconds": float(combo_time),
                }
            )

            print(
                f"\n  FINAL AGGREGATED RESULT — "
                f"{pathway_name}/{model_name}"
            )
            print(
                f"    total runs        = {n_total_runs}"
            )
            print(
                f"    avg Jaccard       = "
                f"{combined_metrics['avg_jaccard']:.4f}"
            )
            print(
                f"    Nogueira stability= "
                f"{combined_metrics['nogueira_stability']:.4f}"
            )
            print(
                f"    stable >=80%      = "
                f"{len(stable_pathways)} pathways"
            )

    # ========================================================
    # SAVE ONLY THE IMPORTANT RESULTS
    # ========================================================

    summary_df = pd.DataFrame(stability_results)

    if summary_df.empty:
        raise RuntimeError(
            "No stability results were produced."
        )

    # Keep the headline table easy to read.
    summary_df = summary_df.sort_values(
        by="nogueira_stability",
        ascending=False,
        na_position="last",
    )

    summary_path = os.path.join(
        STABILITY_DIR,
        "stability_summary_all_methods.csv",
    )

    details_df = pd.DataFrame(pathway_detail_rows)

    if not details_df.empty:
        details_df = details_df.sort_values(
            by=[
                "pathway_method",
                "model",
                "selection_frequency",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

    details_path = os.path.join(
        STABILITY_DIR,
        "stability_pathway_details.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    details_df.to_csv(
        details_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 13 COMPLETE")
    print("=" * 90)

    print("\nHEADLINE STABILITY SUMMARY:")
    display_cols = [
        "pathway_method",
        "model",
        "top_k_used",
        "n_outer_folds_used",
        "n_total_runs",
        "mean_iterations_per_fold",
        "avg_jaccard",
        "nogueira_stability",
        "selection_entropy",
        "n_stable_pathways_ge80pct",
        "best_pathway",
        "best_pathway_selection_frequency",
    ]

    print(
        summary_df[display_cols].to_string(
            index=False
        )
    )

    print("\nSaved only:")
    print(f"  1. {summary_path}")
    print(f"  2. {details_path}")

    print(
        "\n[INFO] No per-combination selection-frequency CSVs "
        "were created."
    )

    return summary_df, details_df


if __name__ == "__main__":
    main()