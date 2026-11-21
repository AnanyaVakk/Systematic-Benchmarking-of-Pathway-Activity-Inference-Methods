
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle

# --------------------------------------------------
# PATHS
# --------------------------------------------------

TCGA_DIR = Path("/path/to/tcga_dir")

OUTPUT_DIR = Path(
    "/path/to/project_dir/"
    "GEO_Final_Validation/models"
)

MODEL_FILE = TCGA_DIR / "best_estimators.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# ORIGINAL TCGA PATHWAY FILES
# --------------------------------------------------

FILES = {
    "ssgsea": "FINAL_Pathway_ssgsea.csv",
    "gsva": "FINAL_Pathway_gsva.csv",
    "plage": "FINAL_Pathway_plage.csv",
    "zscore": "FINAL_Pathway_zscore.csv",
    "progeny": "FINAL_Pathway_PROGENy.csv",
}

EXPECTED_FEATURES = {
    "ssgsea": 50,
    "gsva": 50,
    "plage": 50,
    "zscore": 50,
    "progeny": 14,
}

METADATA_COLUMNS = {
    "file_id",
    "condition",
    "sample",
    "sample_id",
    "sampleid",
    "unnamed: 0",
}

# --------------------------------------------------
# LOAD AND VALIDATE ORIGINAL TCGA FEATURE MATRICES
# --------------------------------------------------

schema = {}

print("=" * 65)
print("TCGA FEATURE SCHEMA GENERATION")
print("=" * 65)

for method, filename in FILES.items():

    path = TCGA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Original TCGA matrix missing: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Empty TCGA matrix: {filename}")

    # Preserve original pathway feature order.
    features = [
        str(col).strip()
        for col in df.columns
        if str(col).strip().lower() not in METADATA_COLUMNS
    ]

    expected = EXPECTED_FEATURES[method]

    if len(features) != expected:
        raise ValueError(
            f"{method}: expected {expected} features, "
            f"found {len(features)}.\n"
            f"Columns: {df.columns.tolist()}"
        )

    if len(features) != len(set(features)):
        raise ValueError(
            f"Duplicate features in {filename}"
        )

    # Validate numeric values and finite data.
    for feature in features:

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        if values.isna().any():
            raise ValueError(
                f"Missing/non-numeric values in "
                f"{method}/{feature}"
            )

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Non-finite values in {method}/{feature}"
            )

    schema[method] = features

    print(
        f"[OK] {method}: {len(features)} features | "
        f"matrix shape: {df.shape}"
    )

# --------------------------------------------------
# CHECK EXISTING TCGA MODEL FILE
# --------------------------------------------------

if not MODEL_FILE.is_file():
    raise FileNotFoundError(
        f"Existing TCGA models missing: {MODEL_FILE}"
    )

with MODEL_FILE.open("rb") as f:
    models = pickle.load(f)

if not isinstance(models, dict):
    raise TypeError(
        "Expected best_estimators.pkl to contain a dictionary."
    )

print("\nExisting TCGA model artifact found.")

# --------------------------------------------------
# SAVE FEATURE SCHEMA
# --------------------------------------------------

output_file = OUTPUT_DIR / "TCGA_feature_schema.json"

with output_file.open("w", encoding="utf-8") as f:
    json.dump(schema, f, indent=4)

# --------------------------------------------------
# VERIFY SAVED JSON
# --------------------------------------------------

with output_file.open("r", encoding="utf-8") as f:
    verified = json.load(f)

if verified != schema:
    raise ValueError("Saved schema verification failed.")

print("\n" + "=" * 65)
print("TCGA FEATURE SCHEMA CREATED SUCCESSFULLY")
print("=" * 65)

print("\nOutput:")
print(output_file)

print("\nFEATURE COUNTS:")

for method, features in verified.items():
    print(f"{method}: {len(features)}")

print("\nExisting TCGA models were not modified.")
print("No training or tuning was performed.")
print("\nREADY FOR FEATURE-ALIGNMENT CHECK.")
