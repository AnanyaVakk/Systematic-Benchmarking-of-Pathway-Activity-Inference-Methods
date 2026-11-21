"""
02 - Add TCGA batch variables (TSS, plate, center) and the Condition label.

Maps each GDC file to its aliquot barcode using the GDC cart JSON, splits the
TCGA barcode into its fields, derives Condition (Tumor / Solid_Tissue_Normal
...) from the sample-type code, and builds composite batch columns.

Inputs:
    <processed-dir>/gdc_sample_sheet_metadata.tsv   (from 01_build_clinical_metadata.py)
    <metadata-dir>/metadata.repository*.json        (GDC cart JSON)
Output:
    <processed-dir>/TCGA_brca_Metadata.tsv          (read by steps 02, 03)

Usage:
    python 02_add_batch_metadata.py
    python 02_add_batch_metadata.py --metadata-dir data/raw/gdc_metadata --processed-dir data/processed
"""
import argparse
import glob
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata-dir", default="data/raw/gdc_metadata",
                        help="Folder holding the GDC cart JSON (metadata.repository*.json).")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Folder with the step 01 output; the result is written here too.")
    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------
    # 1. Load clinical metadata
    # --------------------------------------------------
    meta_path = os.path.join(args.processed_dir, "gdc_sample_sheet_metadata.tsv")
    meta_df = pd.read_csv(meta_path, sep="\t")
    print("Clinical metadata shape:", meta_df.shape)

    # --------------------------------------------------
    # 2. Load GDC cart JSON
    # --------------------------------------------------
    json_files = sorted(glob.glob(os.path.join(args.metadata_dir, "metadata.repository*.json")))
    if not json_files:
        raise FileNotFoundError(
            f"No metadata.repository*.json found in {args.metadata_dir}")
    with open(json_files[0]) as f:
        cart = json.load(f)

    # --------------------------------------------------
    # 3. Build mapping table (case -> aliquot -> file)
    # --------------------------------------------------
    records = []

    for entry in cart:

        file_id = entry["file_id"]
        file_name = entry["file_name"]

        for ent in entry["associated_entities"]:

            aliquot_barcode = ent["entity_submitter_id"]

            # TCGA case barcode = first 12 chars
            case_id = aliquot_barcode[:12]

            records.append({
                "cases.submitter_id": case_id,
                "entity_submitter_id": aliquot_barcode,
                "file_id": file_id,
                "file_name": file_name
            })

    map_df = pd.DataFrame(records).drop_duplicates()
    print("Mapping table shape:", map_df.shape)

    # --------------------------------------------------
    # 4. Merge with clinical metadata
    # --------------------------------------------------
    merged_meta = meta_df.merge(
        map_df,
        on="cases.submitter_id",
        how="left"
    )
    print("Merged metadata shape:", merged_meta.shape)

    # --------------------------------------------------
    # 5. Split TCGA aliquot barcode
    #    Format: TCGA-XX-YYYY-ZZ... (7 fields typically)
    # --------------------------------------------------
    split_cols = merged_meta["entity_submitter_id"].str.split("-", expand=True)

    merged_meta["Project"] = split_cols[0]      # TCGA
    merged_meta["TSS"] = split_cols[1]          # Tissue Source Site
    merged_meta["Participant"] = split_cols[2]
    merged_meta["Sample"] = split_cols[3]
    merged_meta["Portion"] = split_cols[4]
    merged_meta["Plate"] = split_cols[5]
    merged_meta["Center"] = split_cols[6]

    # --------------------------------------------------
    # 6. Derive interpretable variables
    # --------------------------------------------------

    # Sample type code (first 2 digits of sample field)
    merged_meta["Sample_Type_Code"] = merged_meta["Sample"].str[:2]

    # Biological condition
    merged_meta["Condition"] = merged_meta["Sample_Type_Code"].map({
        "01": "Tumor",
        "02": "Recurrent_Tumor",
        "03": "Primary_Blood_Derived_Cancer",
        "10": "Blood_Normal",
        "11": "Solid_Tissue_Normal"
    })

    # Portion + analyte (e.g., 01A, 01R)
    merged_meta["Portion_Analyte"] = merged_meta["Portion"]

    # --------------------------------------------------
    # 7. Composite batch variables
    # --------------------------------------------------
    merged_meta["Batch_TSS_Plate"] = (
        merged_meta["TSS"].astype(str) + "_" +
        merged_meta["Plate"].astype(str)
    )

    merged_meta["Batch_Center_Plate"] = (
        merged_meta["Center"].astype(str) + "_" +
        merged_meta["Plate"].astype(str)
    )

    merged_meta["Batch_TSS_Center"] = (
        merged_meta["TSS"].astype(str) + "_" +
        merged_meta["Center"].astype(str)
    )

    merged_meta["Batch_Full"] = (
        merged_meta["TSS"].astype(str) + "_" +
        merged_meta["Plate"].astype(str) + "_" +
        merged_meta["Center"].astype(str)
    )

    # --------------------------------------------------
    # 8. Final checks
    # --------------------------------------------------
    print("\nPreview with batch columns:\n")
    print(
        merged_meta[
            [
                "entity_submitter_id",
                "file_name",
                "TSS",
                "Plate",
                "Center",
                "Condition",
                "Batch_TSS_Plate",
                "Batch_Full"
            ]
        ].head()
    )

    # --------------------------------------------------
    # 9. Save output (name expected by steps 02 and 03)
    # --------------------------------------------------
    out_path = os.path.join(args.processed_dir, "TCGA_brca_Metadata.tsv")
    merged_meta.to_csv(out_path, sep="\t", index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
