# Calculate GEO pathway scores from TCGA-gene-universe-matched GEO logCPM.
# This script does not normalize counts; script 02 has already applied log2(CPM + 1).
# Algorithm constructors/parameters should match the TCGA pathway-scoring pipeline.

suppressPackageStartupMessages({
  library(GSVA)
  library(msigdbr)
  library(progeny)
})
PROJECT_DIR <- "/path/to/project_dir"
GEO_DIR <- file.path(PROJECT_DIR, "GEO_Final_Validation")
EXPR_FILE <- file.path(GEO_DIR, "Gene_Harmonization", "GEO_common_genes_logCPM_TCGA_matched.csv")
META_FILE <- file.path(GEO_DIR, "input", "GSE58135_metadata_135samples.csv")
OUTPUT_DIR <- file.path(GEO_DIR, "Pathway_Scoring")
dir.create(OUTPUT_DIR, recursive=TRUE, showWarnings=FALSE)
if (!file.exists(EXPR_FILE) || !file.exists(META_FILE)) stop("Run scripts 01 and 02 before pathway scoring.")

expr_df <- read.csv(EXPR_FILE, check.names=FALSE, stringsAsFactors=FALSE)
if (ncol(expr_df) < 2L || names(expr_df)[1] != "Gene_Symbol") stop("Expected Gene_Symbol as first expression column.")
genes <- trimws(as.character(expr_df[[1]]))
if (anyNA(genes) || any(!nzchar(genes)) || anyDuplicated(genes)) stop("Invalid or duplicate gene symbols.")
expr <- as.matrix(data.frame(lapply(expr_df[-1], function(x) as.numeric(as.character(x))), check.names=FALSE))
rownames(expr) <- genes
storage.mode(expr) <- "numeric"
if (any(!is.finite(expr))) stop("Expression matrix contains non-finite values.")

meta <- read.csv(META_FILE, stringsAsFactors=FALSE, check.names=FALSE)
if (!all(c("GSM_ID", "Condition") %in% names(meta))) stop("Metadata requires GSM_ID and Condition.")
meta$GSM_ID <- trimws(as.character(meta$GSM_ID))
if (nrow(meta) != 135L || sum(meta$Condition=="Tumor") != 84L || sum(meta$Condition=="Normal") != 51L)
  stop("Expected 135 samples: 84 Tumor and 51 Normal.")
if (anyDuplicated(meta$GSM_ID) || !setequal(colnames(expr), meta$GSM_ID)) stop("Expression/metadata sample IDs do not match exactly.")
expr <- expr[, meta$GSM_ID, drop=FALSE]

cat("Input: ", nrow(expr), " genes x ", ncol(expr), " samples\n", sep="")
cat("Input scale: GEO log2(CPM + 1); TMM not applied\n")
msig <- msigdbr(species="Homo sapiens", collection="H")
gene_sets <- lapply(split(msig$gene_symbol, msig$gs_name), unique)
overlap_n <- vapply(gene_sets, function(gs) length(intersect(gs, rownames(expr))), integer(1))
if (any(overlap_n == 0L)) stop("At least one Hallmark gene set has zero overlap with GEO expression.")

write_sample_rows <- function(score_matrix, filename, transpose=TRUE) {
  score_matrix <- as.matrix(score_matrix)
  if (transpose) score_matrix <- t(score_matrix)
  if (is.null(rownames(score_matrix)) || !identical(rownames(score_matrix), meta$GSM_ID))
    stop("Output sample IDs/order do not match metadata for ", filename)
  if (any(!is.finite(score_matrix))) stop("Non-finite scores in ", filename)
  out <- data.frame(File_ID=rownames(score_matrix), score_matrix, check.names=FALSE, row.names=NULL)
  write.csv(out, file.path(OUTPUT_DIR, filename), row.names=FALSE, quote=FALSE)
  cat("[OK] ", filename, " : ", nrow(out), " samples x ", ncol(out)-1L, " pathways\n", sep="")
}

cat("Running ssGSEA...\n")
ssgsea_res <- gsva(ssgseaParam(expr, gene_sets))
write_sample_rows(ssgsea_res, "GEO_Pathway_ssgsea.csv")
cat("Running GSVA...\n")
gsva_res <- gsva(gsvaParam(expr, gene_sets))
write_sample_rows(gsva_res, "GEO_Pathway_gsva.csv")
cat("Running PLAGE...\n")
plage_res <- gsva(plageParam(expr, gene_sets))
write_sample_rows(plage_res, "GEO_Pathway_plage.csv")

# Retains the custom within-sample pathway z-score definition from the prior script.
cat("Running custom Z-score...\n")
zscore_res <- vapply(gene_sets, function(gs) {
  common <- intersect(gs, rownames(expr))
  if (length(common) > 1L) {
    vals <- expr[common, , drop=FALSE]
    sds <- apply(vals, 2, sd)
    means <- colMeans(vals)
    ifelse(is.finite(sds) & sds > 0, means / sds, NA_real_)
  } else rep(NA_real_, ncol(expr))
}, numeric(ncol(expr)))
rownames(zscore_res) <- meta$GSM_ID
write_sample_rows(zscore_res, "GEO_Pathway_zscore.csv", transpose=FALSE)

cat("Running PROGENy...\n")
progeny_res <- progeny(expr, scale=TRUE, organism="Human", top=500)
# PROGENy normally returns samples x pathways.
write_sample_rows(progeny_res, "GEO_Pathway_PROGENy.csv", transpose=FALSE)

provenance <- data.frame(
  item=c("expression_input", "normalization", "TMM", "samples", "tumor", "normal",
         "hallmark_pathways", "PLAGE_sign_orientation"),
  value=c(EXPR_FILE, "log2(CPM + 1), ordinary CPM", "NOT APPLIED", ncol(expr),
          sum(meta$Condition=="Tumor"), sum(meta$Condition=="Normal"), length(gene_sets),
          "Independent-cohort SVD sign ambiguity not anchored; interpret PLAGE cautiously")
)
write.csv(provenance, file.path(OUTPUT_DIR, "GEO_pathway_scoring_provenance.csv"), row.names=FALSE)
cat("GEO pathway scoring completed: ", OUTPUT_DIR, "\n", sep="")
