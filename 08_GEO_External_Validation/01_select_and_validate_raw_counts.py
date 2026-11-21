"""Select and validate the curated 135 GSE58135 raw-count samples."""
import numpy as np
import pandas as pd
from config import *

print(f'Reading raw counts: {RAW_COUNTS_FILE}')
if not RAW_COUNTS_FILE.is_file():
    raise FileNotFoundError(f'Raw-count TSV not found: {RAW_COUNTS_FILE}')

raw = pd.read_csv(RAW_COUNTS_FILE, sep='\t', index_col=0, low_memory=False)
raw.index = raw.index.astype(str).str.strip()
raw.columns = raw.columns.astype(str).str.strip()
if raw.index.has_duplicates:
    raise ValueError('Duplicate GeneID rows found. Resolve duplicates before continuing.')

missing = [sid for sid in SELECTED_IDS if sid not in raw.columns]
if missing:
    raise ValueError(f'Missing expected GSM columns ({len(missing)}): {missing[:20]}')

counts = raw.loc[:, SELECTED_IDS].apply(pd.to_numeric, errors='coerce')
if counts.isna().any().any():
    raise ValueError(f'Non-numeric/missing count cells: {int(counts.isna().sum().sum())}')
arr = counts.to_numpy(dtype=float)
if not np.isfinite(arr).all() or (arr < 0).any():
    raise ValueError('Counts contain non-finite or negative values.')
if not np.allclose(arr, np.rint(arr), atol=1e-8):
    raise ValueError('Input values are not integer-like raw counts.')
counts = counts.round().astype('int64')
counts.index.name = 'GeneID'

count_path = COUNT_DIR / 'GSE58135_raw_counts_135samples.csv'
counts.to_csv(count_path)
meta = pd.DataFrame({
    'GSM_ID': SELECTED_IDS,
    'Condition': ['Tumor' if sid in set(TUMOR_IDS) else 'Normal' for sid in SELECTED_IDS]
})
vc = meta.Condition.value_counts().to_dict()
if len(meta) != EXPECTED_SAMPLES or vc.get('Tumor', 0) != EXPECTED_TUMOR or vc.get('Normal', 0) != EXPECTED_NORMAL:
    raise AssertionError(f'Unexpected cohort composition: {vc}')
meta.to_csv(INPUT_DIR / 'GSE58135_metadata_135samples.csv', index=False)

print(f'Raw matrix: {raw.shape[0]} genes x {raw.shape[1]} samples')
print(f'Selected matrix: {counts.shape[0]} genes x {counts.shape[1]} samples')
print(f'Metadata: {vc}')
print(f'Saved counts: {count_path}')
