#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(optparse)
})

# --- Define command-line options ---
option_list <- list(
  make_option(c("-i", "--input"), type = "character", help = "Indicator BED file", metavar = "FILE"),
  make_option(c("-m", "--metadata"), type = "character", help = "Metadata CSV file", metavar = "FILE"),
  make_option(c("-o", "--output"), type = "character", default = "dataset_1kb_300bp_S3.bed",
              help = "Output BED file [default: %default]", metavar = "FILE"),
  make_option(c("-t", "--threshold"), type = "double", default = 0.7,
              help = "Threshold for cCRE indicator [default: %default]"),
  make_option(c("-r", "--ratio"), type = "double", default = 4,
              help = "Negative-to-positive sampling ratio [default: %default]")
)

# --- Parse arguments ---
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# --- Check required arguments ---
if (is.null(opt$input) || is.null(opt$metadata)) {
  print_help(opt_parser)
  stop("Error: --input and --metadata are required.", call. = FALSE)
}

# --- Read metadata and set column names ---
metadata <- read.csv(opt$metadata, header = TRUE)
context_ids <- metadata$context
column_names <- c("chr", "start", "end", "name", context_ids)

# --- Read and clean indicator data ---
data <- read.delim(opt$input, header = FALSE)
data <- data[complete.cases(data), ]

# --- Define signal columns and indicator column ---
num_signals <- length(context_ids)
signal_cols <- 5:(4 + num_signals)
indicator_col <- 4 + num_signals + 1

# --- Quantile-based filtering ---
upper_limits <- sapply(data[, signal_cols], quantile, probs = 0.995)
lower_limits <- sapply(data[, signal_cols], quantile, probs = 0.1)

for (j in seq_along(signal_cols)) {
  idx <- signal_cols[j]
  data <- data[data[, idx] < upper_limits[j], ]
  data <- data[data[, idx] > lower_limits[j], ]
}

# --- Log2 transform signal values ---
data[, signal_cols] <- log2(data[, signal_cols])

# --- Split positive and negative samples ---
positive_samples <- data[data[[indicator_col]] >= opt$threshold, ]
negative_samples <- data[data[[indicator_col]] < opt$threshold, ]

# --- Downsample negatives based on ratio ---
num_neg <- as.integer(nrow(positive_samples) / opt$ratio)
sampled_negatives <- negative_samples %>%
  slice_sample(n = min(num_neg, nrow(negative_samples)))

# --- Combine and drop indicator column ---
output_data <- bind_rows(positive_samples, sampled_negatives) %>%
  select(-all_of(indicator_col))
colnames(output_data) <- column_names

# --- Write output ---
write.table(
  output_data,
  file = opt$output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = FALSE
)

