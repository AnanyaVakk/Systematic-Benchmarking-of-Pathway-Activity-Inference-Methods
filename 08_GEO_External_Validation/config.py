"""Shared paths and cohort definitions for GSE58135 external validation."""
from pathlib import Path

PROJECT_DIR = Path('/path/to/project_dir')
GEO_DIR = PROJECT_DIR / 'GEO_Final_Validation'

RAW_COUNTS_FILE = PROJECT_DIR / 'GSE58135_raw_counts_GRCh38.p13_NCBI.tsv'
SOFT_FILE = PROJECT_DIR / 'GSE58135_family.soft'

# Frozen TCGA artifacts (read-only during external validation).
TCGA_DIR = Path('/path/to/tcga_dir')
TCGA_LOGCPM_FILE = TCGA_DIR / 'logCPM_matrix.csv'
FROZEN_MODELS_FILE = TCGA_DIR / 'best_estimators.pkl'
FEATURE_SCHEMA_FILE = GEO_DIR / 'models' / 'TCGA_feature_schema.json'

INPUT_DIR = GEO_DIR / 'input'
COUNT_DIR = GEO_DIR / 'counts'
NORM_DIR = GEO_DIR / 'normalized'
NORMALIZED_DIR = NORM_DIR
QC_DIR = GEO_DIR / 'QC'
HARM_DIR = GEO_DIR / 'Gene_Harmonization'
PATHWAY_DIR = GEO_DIR / 'Pathway_Scoring'
RESULT_DIR = GEO_DIR / 'External_Validation_Results'

TUMOR_IDS = ([f'GSM{i}' for i in range(1401676, 1401718)] +
             [f'GSM{i}' for i in range(1401718, 1401760)])
NORMAL_IDS = ([f'GSM{i}' for i in range(1401760, 1401790)] +
              [f'GSM{i}' for i in range(1401795, 1401816)])
SELECTED_IDS = TUMOR_IDS + NORMAL_IDS

EXPECTED_SAMPLES = 135
EXPECTED_TUMOR = 84
EXPECTED_NORMAL = 51
POSITIVE_LABEL = 1
THRESHOLD = 0.50

for folder in (INPUT_DIR, COUNT_DIR, NORM_DIR, QC_DIR, HARM_DIR,
               PATHWAY_DIR, RESULT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

if len(TUMOR_IDS) != EXPECTED_TUMOR or len(NORMAL_IDS) != EXPECTED_NORMAL:
    raise ValueError('Configured GEO sample ID counts do not match expected cohort sizes.')
if len(set(SELECTED_IDS)) != EXPECTED_SAMPLES:
    raise ValueError('Configured GEO sample IDs are not unique or do not total 135.')
