
# ============================================================
# GSE58135 EXTERNAL VALIDATION
# TCGA-CONSISTENT ORDINARY CPM NORMALIZATION
#
# Formula:
# logCPM = log2(CPM + 1)
#
# NO TMM
# NO calcNormFactors()
# NO prior.count transformation
#
# Gene harmonization:
# - Entrez ID -> gene symbol
# - Aggregate duplicate symbols using raw counts
# - Preserve TCGA reference gene order
# - Identify missing and common genes
# - Do NOT independently filter GEO by variance
# ============================================================


suppressPackageStartupMessages({
  library(edgeR)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
})


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR <- "/path/to/project_dir"

GEO_DIR <- file.path(PROJECT_DIR, "GEO_Final_Validation")

TCGA_LOGCPM_FILE <- "/path/to/tcga_dir/logCPM_matrix.csv"

COUNT_FILE <- file.path(
  GEO_DIR, "counts",
  "GSE58135_raw_counts_135samples.csv"
)

META_FILE <- file.path(
  GEO_DIR, "input",
  "GSE58135_metadata_135samples.csv"
)

NORMALIZED_DIR <- file.path(GEO_DIR, "normalized")

HARM_DIR <- file.path(GEO_DIR, "Gene_Harmonization")

dir.create(NORMALIZED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(HARM_DIR, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------
# 2. CHECK INPUT FILES
# ------------------------------------------------------------

required <- c(
  COUNT_FILE,
  META_FILE,
  TCGA_LOGCPM_FILE
)

missing_files <- required[!file.exists(required)]

if (length(missing_files) > 0) {
  stop(
    "Required files missing:\n",
    paste(missing_files, collapse = "\n")
  )
}


# ------------------------------------------------------------
# 3. READ RAW GEO COUNTS AND METADATA
# ------------------------------------------------------------

cat("\nReading GEO raw counts...\n")

counts_df <- read.csv(
  COUNT_FILE,
  row.names = 1,
  check.names = FALSE
)

meta <- read.csv(
  META_FILE,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

if (!all(c("GSM_ID", "Condition") %in% names(meta))) {
  stop("Metadata must contain GSM_ID and Condition.")
}

meta$GSM_ID <- trimws(as.character(meta$GSM_ID))
meta$Condition <- trimws(as.character(meta$Condition))

if (anyDuplicated(meta$GSM_ID)) {
  stop("Duplicate GSM IDs in metadata.")
}

if (any(!meta$Condition %in% c("Tumor", "Normal"))) {
  stop("Unexpected condition labels.")
}

if (
  nrow(meta) != 135L ||
  sum(meta$Condition == "Tumor") != 84L ||
  sum(meta$Condition == "Normal") != 51L
) {
  stop("Expected 135 samples: 84 Tumor and 51 Normal.")
}

if (!all(meta$GSM_ID %in% colnames(counts_df))) {
  stop("Some metadata GSM IDs are absent from raw counts.")
}

# Arrange samples according to metadata order.

counts_df <- counts_df[
  ,
  meta$GSM_ID,
  drop = FALSE
]

counts <- as.matrix(counts_df)

storage.mode(counts) <- "numeric"

if (any(!is.finite(counts)) || any(counts < 0)) {
  stop("Invalid raw count values.")
}

if (any(abs(counts - round(counts)) > 1e-8)) {
  stop("Input must contain integer-valued raw counts.")
}

storage.mode(counts) <- "integer"


# ------------------------------------------------------------
# 4. MAP ENTREZ IDS TO GENE SYMBOLS
# ------------------------------------------------------------

cat("\nMapping Entrez IDs to gene symbols...\n")

# Remove optional version suffixes, e.g. 1234.1

entrez <- sub(
  "\\.\\d+$",
  "",
  trimws(rownames(counts))
)

if (anyNA(entrez) || any(!nzchar(entrez))) {
  stop("Blank or invalid Entrez IDs found.")
}

map <- AnnotationDbi::mapIds(
  org.Hs.eg.db,
  keys = unique(entrez),
  keytype = "ENTREZID",
  column = "SYMBOL",
  multiVals = "first"
)

symbols <- unname(map[entrez])

keep <- !is.na(symbols) & nzchar(symbols)

counts_mapped <- counts[
  keep,
  ,
  drop = FALSE
]

rownames(counts_mapped) <- symbols[keep]

if (nrow(counts_mapped) == 0) {
  stop("No genes successfully mapped.")
}


# ------------------------------------------------------------
# 5. AGGREGATE DUPLICATE GENE SYMBOLS
# ------------------------------------------------------------

cat("\nAggregating duplicate gene symbols...\n")

# IMPORTANT:
# Sum raw counts before calculating CPM.

counts_symbol <- rowsum(
  counts_mapped,
  group = rownames(counts_mapped),
  reorder = FALSE
)

storage.mode(counts_symbol) <- "integer"

if (anyDuplicated(rownames(counts_symbol))) {
  stop("Duplicate symbols remain after aggregation.")
}


# ------------------------------------------------------------
# 6. ORDINARY CPM NORMALIZATION
# ------------------------------------------------------------

cat("\nCalculating ordinary CPM...\n")

dge <- DGEList(
  counts = counts_symbol
)

# Explicitly disable normalized library sizes.

cpm_matrix <- cpm(
  dge,
  normalized.lib.sizes = FALSE,
  log = FALSE
)

# TCGA-consistent transformation.

log_cpm <- log2(cpm_matrix + 1)

if (any(!is.finite(log_cpm))) {
  stop("Non-finite values generated during logCPM.")
}

cat("Normalization completed.\n")


# ------------------------------------------------------------
# 7. SAVE COMPLETE GEO NORMALIZED MATRIX
# ------------------------------------------------------------

geo_normalized_file <- file.path(
  NORMALIZED_DIR,
  "GSE58135_logCPM_gene_symbols_TCGA_formula.csv"
)

write.csv(
  data.frame(
    Gene_Symbol = rownames(log_cpm),
    log_cpm,
    check.names = FALSE
  ),
  geo_normalized_file,
  row.names = FALSE,
  quote = FALSE
)


# ------------------------------------------------------------
# 8. READ TCGA REFERENCE GENE UNIVERSE
# ------------------------------------------------------------

cat("\nReading TCGA reference gene universe...\n")

ref_df <- read.csv(
  TCGA_LOGCPM_FILE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (ncol(ref_df) < 2L) {
  stop("TCGA reference must contain gene IDs and sample columns.")
}

ref_genes <- trimws(
  as.character(ref_df[[1L]])
)

if (anyNA(ref_genes) || any(!nzchar(ref_genes))) {
  stop("Invalid TCGA gene symbols.")
}

if (anyDuplicated(ref_genes)) {
  stop("Duplicate genes in TCGA reference.")
}


# ------------------------------------------------------------
# 9. IDENTIFY COMMON AND MISSING GENES
# ------------------------------------------------------------

cat("\nHarmonizing GEO and TCGA genes...\n")

geo_genes <- rownames(log_cpm)

# Preserve the exact TCGA reference order.

common_genes <- ref_genes[
  ref_genes %in% geo_genes
]

missing_geo <- ref_genes[
  !ref_genes %in% geo_genes
]

geo_extra <- setdiff(
  geo_genes,
  ref_genes
)

if (length(common_genes) == 0) {
  stop("No common genes between TCGA and GEO.")
}

cat("TCGA reference genes:", length(ref_genes), "\n")
cat("GEO mapped genes:", length(geo_genes), "\n")
cat("Common genes:", length(common_genes), "\n")
cat("Missing GEO genes:", length(missing_geo), "\n")
cat("GEO extra genes:", length(geo_extra), "\n")


# ------------------------------------------------------------
# 10. CREATE TCGA-ORDERED COMMON GENE MATRIX
# ------------------------------------------------------------

matched <- log_cpm[
  common_genes,
  meta$GSM_ID,
  drop = FALSE
]

if (!identical(rownames(matched), common_genes)) {
  stop("Gene ordering mismatch.")
}

if (!identical(colnames(matched), meta$GSM_ID)) {
  stop("Sample ordering mismatch.")
}

matched_file <- file.path(
  HARM_DIR,
  "GEO_common_genes_logCPM_TCGA_matched.csv"
)

write.csv(
  data.frame(
    Gene_Symbol = rownames(matched),
    matched,
    check.names = FALSE
  ),
  matched_file,
  row.names = FALSE,
  quote = FALSE
)


# ------------------------------------------------------------
# 11. SAVE MISSING GENES
# ------------------------------------------------------------

write.csv(
  data.frame(
    Missing_TCGA_Gene = missing_geo
  ),
  file.path(
    HARM_DIR,
    "TCGA_genes_missing_from_GSE58135.csv"
  ),
  row.names = FALSE,
  quote = FALSE
)


# ------------------------------------------------------------
# 12. SAVE GEO EXTRA GENES
# ------------------------------------------------------------

write.csv(
  data.frame(
    GEO_extra_gene = geo_extra
  ),
  file.path(
    HARM_DIR,
    "GSE58135_extra_genes_not_in_TCGA.csv"
  ),
  row.names = FALSE,
  quote = FALSE
)


# ------------------------------------------------------------
# 13. SAMPLE METADATA FOR EXTERNAL VALIDATION
# ------------------------------------------------------------

write.csv(
  meta,
  file.path(
    HARM_DIR,
    "GSE58135_validation_metadata.csv"
  ),
  row.names = FALSE
)


# ------------------------------------------------------------
# 14. QUALITY CONTROL REPORT
# ------------------------------------------------------------

qc <- data.frame(
  metric = c(
    "raw_input_genes",
    "mapped_rows",
    "unique_gene_symbols",
    "TCGA_reference_genes",
    "GEO_mapped_genes",
    "common_genes",
    "missing_TCGA_genes_in_GEO",
    "GEO_extra_genes",
    "GEO_samples",
    "Tumor_samples",
    "Normal_samples",
    "TMM_applied",
    "zero_variance_filter_applied_to_GEO",
    "normalization_formula"
  ),

  value = as.character(c(
    nrow(counts),
    nrow(counts_mapped),
    nrow(counts_symbol),
    length(ref_genes),
    length(geo_genes),
    length(common_genes),
    length(missing_geo),
    length(geo_extra),
    ncol(matched),
    sum(meta$Condition == "Tumor"),
    sum(meta$Condition == "Normal"),
    "NO",
    "NO",
    "log2(CPM + 1)"
  ))
)

write.csv(
  qc,
  file.path(
    NORMALIZED_DIR,
    "GEO_TCGA_matched_preprocessing_QC.csv"
  ),
  row.names = FALSE
)


# ------------------------------------------------------------
# 15. FINAL VALIDATION
# ------------------------------------------------------------

cat("\n========================================\n")
cat("GSE58135 PREPROCESSING COMPLETED\n")
cat("========================================\n")

cat("Raw input genes:", nrow(counts), "\n")

cat("Mapped rows:", nrow(counts_mapped), "\n")

cat("Unique gene symbols:", nrow(counts_symbol), "\n")

cat("TCGA reference genes:", length(ref_genes), "\n")

cat("Common genes:", nrow(matched), "\n")

cat("Missing genes:", length(missing_geo), "\n")

cat("GEO samples:", ncol(matched), "\n")

cat("Tumor samples:", sum(meta$Condition == "Tumor"), "\n")

cat("Normal samples:", sum(meta$Condition == "Normal"), "\n")

cat("TMM applied: NO\n")

cat("Normalization: log2(CPM + 1)\n")

cat("\nOutput files saved successfully.\n")

cat("\nNormalized directory:\n", NORMALIZED_DIR, "\n")

cat("\nGene harmonization directory:\n", HARM_DIR, "\n")

cat("\n========================================\n")
