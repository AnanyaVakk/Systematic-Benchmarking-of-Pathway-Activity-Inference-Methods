"""
01 - Build the clinical metadata table for the TCGA samples.

Merges the GDC sample sheet with the clinical and exposure tables, keeps the
columns used downstream, and keeps one row per File ID (the row with the most
non-missing values).

Inputs  (in --metadata-dir, default data/raw/gdc_metadata):
    gdc_sample_sheet*.tsv   GDC sample sheet
    clinical.tsv            GDC clinical table
    exposure.tsv            GDC exposure table
Output  (in --output-dir, default data/processed):
    gdc_sample_sheet_metadata.tsv

Usage:
    python 01_build_clinical_metadata.py
    python 01_build_clinical_metadata.py --metadata-dir path/to/gdc_metadata --output-dir data/processed
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata-dir", default="data/raw/gdc_metadata",
                        help="Folder with the GDC sample sheet, clinical.tsv and exposure.tsv.")
    parser.add_argument("--sample-sheet", default=None,
                        help="Sample sheet path. Default: the gdc_sample_sheet*.tsv in --metadata-dir.")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Folder for the output metadata table.")
    return parser.parse_args()


def find_sample_sheet(metadata_dir):
    matches = [
        p for p in sorted(glob.glob(os.path.join(metadata_dir, "gdc_sample_sheet*.tsv")))
        if not p.endswith("_metadata.tsv")
    ]
    if not matches:
        raise FileNotFoundError(f"No gdc_sample_sheet*.tsv found in {metadata_dir}")
    return matches[0]


def main():
    args = parse_args()

    sample_sheet_path = args.sample_sheet or find_sample_sheet(args.metadata_dir)
    print("Sample sheet:", sample_sheet_path)

    brca_samplesheet = pd.read_csv(sample_sheet_path, sep="\t")
    brca_clinical = pd.read_csv(os.path.join(args.metadata_dir, "clinical.tsv"), sep="\t")
    brca_exposure = pd.read_csv(os.path.join(args.metadata_dir, "exposure.tsv"), sep="\t")

    df_merged = brca_samplesheet.merge(
        brca_clinical,
        left_on="Case ID",
        right_on="cases.submitter_id",
        how="left"
    )

    df_merged = df_merged.merge(
        brca_exposure,
        on="cases.submitter_id",
        how="left"
    )

    cols_to_keep = [
        # Sample sheet
        "File ID",
        "Tissue Type",
        "Case ID",

        # Clinical
        "cases.submitter_id",
        "cases.disease_type",
        "demographic.age_at_index",
        "demographic.country_of_residence_at_enrollment",
        "demographic.days_to_death",
        "demographic.gender",
        "demographic.vital_status",
        "diagnoses.age_at_diagnosis",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.classification_of_tumor",

        # Exposure
        "exposures.tobacco_smoking_status"
    ]

    brca_final_df = df_merged[cols_to_keep]

    df = brca_final_df.copy()
    df = df.replace(["'--", ""], np.nan)

    # Keep the most complete row for each File ID
    cols_for_completeness = [c for c in df.columns if c != "File ID"]
    df["non_null_count"] = df[cols_for_completeness].notna().sum(axis=1)

    df_sorted = df.sort_values(
        by=["File ID", "non_null_count"],
        ascending=[True, False]
    )

    df_dedup = (
        df_sorted
        .drop_duplicates(subset="File ID", keep="first")
        .drop(columns="non_null_count")
    )

    brca_final_df = df_dedup

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "gdc_sample_sheet_metadata.tsv")
    brca_final_df.to_csv(out_path, sep="\t", index=False)

    print(brca_final_df.head())
    print("Shape:", brca_final_df.shape)
    print("Missing values per column:")
    print(brca_final_df.isnull().sum())
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
