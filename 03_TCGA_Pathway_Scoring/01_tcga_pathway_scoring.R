# ==========================================
# 03 - TCGA pathway activity scoring (GSVA, ssGSEA, PLAGE, Z-score, PROGENy)
#
# Inputs  (in the data directory):
#   merged_count_matrix_UUID.csv, TCGA_brca_Metadata.tsv
# Outputs (written to the same directory):
#   FINAL_Pathway_{gsva,ssgsea,plage,zscore,PROGENy}.csv   (used by steps 04-07)
#   TCGA_*.rds reference files                              (used for GEO external validation)
#
# Usage:
#   Rscript 01_tcga_pathway_scoring.R [data_dir]      # default data_dir = data/processed
# ==========================================
# ==========================================
# 1. LOAD LIBRARIES
# ==========================================
library(edgeR)
library(msigdbr)
library(dplyr)
library(GSVA)

if ("progeny" %in% rownames(installed.packages())) {
  library(progeny)
}

# ==========================================
# 2. SET WORKING DIRECTORY
# ==========================================
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "data/processed"
if (!dir.exists(data_dir)) stop(paste("Data directory not found:", data_dir))
setwd(data_dir)

# ==========================================
# 3. LOAD COUNTS
# ==========================================
counts_raw <- read.delim("merged_count_matrix_UUID.csv", check.names = FALSE)

# Separate gene names
gene_names <- counts_raw[[1]]

# Extract numeric part ONLY
counts <- counts_raw[,-1]

# Convert column by column safely
counts <- as.data.frame(lapply(counts, function(col) {
  suppressWarnings(as.numeric(col))
}))

# Check how bad NA is
cat("NA count:", sum(is.na(counts)), "\n")

# Replace NA with 0 (critical)
counts[is.na(counts)] <- 0

# Convert to matrix
counts <- as.matrix(counts)

# Assign rownames
rownames(counts) <- gene_names

# FINAL CHECKS
cat("Dim:", dim(counts), "\n")
cat("Total sum:", sum(counts), "\n")

# ==========================================
# 4. LOAD METADATA
# ==========================================
meta <- read.delim("TCGA_brca_Metadata.tsv")
colnames(meta)[colnames(meta) == "File ID"] <- "File_ID"
meta <- meta[!duplicated(meta$File_ID), ]

# ==========================================
# 5. TMM NORMALIZATION + logCPM
# ==========================================
dge <- DGEList(counts = counts)

dge <- calcNormFactors(dge, method = "TMM")

log_cpm <- cpm(dge, log = TRUE, prior.count = 1)

# Remove constant genes
log_cpm <- log_cpm[apply(log_cpm, 1, var) != 0, ]

expr <- log_cpm

cat("Expression matrix:", dim(expr), "\n")

# ==========================================
# 6. LOAD PATHWAYS
# ==========================================
msig <- msigdbr(species = "Homo sapiens", category = "H")
gene_sets <- split(msig$gene_symbol, msig$gs_name)

# ==========================================
# 7. GSVA PARAM OBJECTS (EXPLICIT)
# ==========================================

# ---- GSVA ----
gsvaPar <- gsvaParam(
  exprData = expr,
  geneSets = gene_sets,
  kcdf = "Gaussian",
  minSize = 10,
  maxSize = 500,
  maxDiff = TRUE,
  absRanking = FALSE
)

# ---- ssGSEA ----
ssgseaPar <- ssgseaParam(
  exprData = expr,
  geneSets = gene_sets,
  minSize = 10,
  maxSize = 500,
  alpha = 0.25,
  normalize = TRUE
)

# ---- PLAGE ----
plagePar <- plageParam(
  exprData = expr,
  geneSets = gene_sets,
  minSize = 10,
  maxSize = 500
)

# ---- Z-SCORE ----
zPar <- zscoreParam(
  exprData = expr,
  geneSets = gene_sets,
  minSize = 10,
  maxSize = 500
)

# ==========================================
# 8. RUN METHODS (ONE BY ONE)
# ==========================================

run_and_save <- function(res, method_name) {
  
  df <- as.data.frame(t(as.matrix(res)))
  df$File_ID <- rownames(df)
  
  df$Condition <- meta$Condition[
    match(df$File_ID, meta$File_ID)
  ]
  
  filename <- paste0("FINAL_Pathway_", method_name, ".csv")
  write.csv(df, filename, row.names = FALSE)
  
  cat(method_name, "completed\n")
}

# ---- GSVA ----
cat("Running GSVA\n")
gsva_res <- gsva(gsvaPar)
run_and_save(gsva_res, "gsva")

# ---- ssGSEA ----
cat("Running ssGSEA\n")
ssgsea_res <- gsva(ssgseaPar)
run_and_save(ssgsea_res, "ssgsea")

# ---- PLAGE ----
cat("Running PLAGE\n")
plage_res <- gsva(plagePar)
run_and_save(plage_res, "plage") 


# ============================================================
# 10. EXPORT TCGA REFERENCE FOR GEO EXTERNAL VALIDATION
# ============================================================

cat("\nExporting TCGA reference files...\n")

# Save exact gene sets used for TCGA scoring
saveRDS(
  gene_sets,
  "TCGA_Hallmark_gene_sets.rds"
)

# Save the gene universe after TCGA filtering
saveRDS(
  rownames(expr),
  "TCGA_gene_universe.rds"
)

# Save TCGA pathway names and original PLAGE scores
saveRDS(
  rownames(plage_res),
  "TCGA_pathway_names.rds"
)

saveRDS(
  plage_res,
  "TCGA_PLAGE_reference_scores.rds"
)

# ------------------------------------------------------------
# Generate fixed-reference PLAGE loadings
# ------------------------------------------------------------

plage_reference <- list()

for (pw in rownames(plage_res)) {
  
  # Genes belonging to this pathway and present in TCGA
  genes <- intersect(
    rownames(expr),
    gene_sets[[pw]]
  )
  
  # Apply the same pathway size limits
  if (length(genes) < 10 || length(genes) > 500) {
    next
  }
  
  # TCGA expression for this pathway
  x <- expr[genes, , drop = FALSE]
  
  # Calculate gene-wise TCGA means and SDs
  gene_means <- rowMeans(x)
  gene_sds <- apply(x, 1, sd)
  
  # Remove genes with invalid or zero SD
  valid <- is.finite(gene_sds) & gene_sds > 0
  
  x <- x[valid, , drop = FALSE]
  gene_means <- gene_means[valid]
  gene_sds <- gene_sds[valid]
  
  if (nrow(x) < 10) {
    next
  }
  
  # Standardise using TCGA statistics
  z <- sweep(x, 1, gene_means, "-")
  z <- sweep(z, 1, gene_sds, "/")
  
  # Perform SVD on TCGA pathway expression
  sv <- svd(z, nu = 1, nv = 1)
  
  # First right singular vector = TCGA sample scores
  tcga_scores <- sv$v[, 1]
  
  # Align sign with the original GSVA PLAGE result
  original_scores <- as.numeric(plage_res[pw, colnames(expr)])
  
  sign_correlation <- cor(
    tcga_scores,
    original_scores,
    use = "complete.obs"
  )
  
  sign_multiplier <- ifelse(
    sign_correlation < 0,
    -1,
    1
  )
  
  # Store fixed gene loadings and TCGA reference statistics
  plage_reference[[pw]] <- list(
    genes = rownames(x),
    gene_means = gene_means,
    gene_sds = gene_sds,
    loadings = sv$u[, 1] * sign_multiplier,
    singular_value = sv$d[1],
    sign_multiplier = sign_multiplier,
    correlation_with_original = sign_correlation
  )
}

# Save PLAGE reference
saveRDS(
  plage_reference,
  "TCGA_PLAGE_fixed_reference.rds"
)

cat(
  "PLAGE reference pathways:",
  length(plage_reference),
  "\n"
)

cat("TCGA reference export completed.\n")


# ---- Z-SCORE ----
cat("Running Z-score\n")
zscore_res <- gsva(zPar)
run_and_save(zscore_res, "zscore")

# ==========================================
# 9. PROGENy (OPTIONAL)
# ==========================================
if ("progeny" %in% rownames(installed.packages())) {
  
  cat("Running PROGENy\n")
  
  progeny_res <- progeny(
    expr,
    scale = FALSE,
    organism = "Human",
    top = 500
  )
  
  progeny_df <- as.data.frame(progeny_res)
  progeny_df$File_ID <- rownames(progeny_df)
  
  progeny_df$Condition <- meta$Condition[
    match(progeny_df$File_ID, meta$File_ID)
  ]
  
  write.csv(progeny_df, "FINAL_Pathway_PROGENy.csv", row.names = FALSE)
}

# ==========================================
# DONE
# ==========================================
cat("ALL METHODS COMPLETED SUCCESSFULLY\n")