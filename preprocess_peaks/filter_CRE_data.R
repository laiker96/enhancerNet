#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(optparse)
})

# --- Define CLI options ---
option_list <- list(
  make_option(c("-c", "--cre_file"), type = "character", help = "Input BED file with CRE data", metavar = "FILE"),
  make_option(c("-m", "--metadata"), type = "character", help = "Metadata CSV with SRA to context mapping", metavar = "FILE"),
  make_option(c("-o", "--output"), type = "character", default = "dataset_1kb.bed", help = "Output BED file [default: %default]"),
  make_option(c("-r", "--ratio"), type = "double", default = 6, help = "Negative:positive downsampling ratio [default: %default]")
)

# --- Parse options ---
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# --- Check required inputs ---
if (is.null(opt$cre_file) || is.null(opt$metadata)) {
  print_help(opt_parser)
  stop("Error: --cre_file, and --metadata are required.", call. = FALSE)
}

# --- Read and map header names ---
metadata_df <- read.csv(opt$metadata, header = TRUE)
context_names <- metadata_df$context
col_names <- c("chr", "start", "end", "name", context_names)

# --- Read CRE data ---
data <- read.delim(opt$cre_file, header = FALSE)
data <- data[complete.cases(data), ]

# --- Define columns ---
signal_cols <- setdiff(seq_along(data), c(1:4, 14))  # exclude chr/start/end/name + indicator

# --- Split into positives and negatives ---
positive_samples <- data %>% filter(rowSums(across(all_of(signal_cols))) > 0)
negative_samples <- data %>% filter(rowSums(across(all_of(signal_cols))) == 0 & V14 == 0)

# --- Downsample negatives ---
num_neg <- min(nrow(negative_samples), floor(nrow(positive_samples) / opt$ratio))
sampled_negatives <- slice_sample(negative_samples, n = num_neg)

# --- Combine and rename ---
output_data <- bind_rows(positive_samples, sampled_negatives) %>%
  select(-V14)  # remove indicator column
colnames(output_data) <- col_names

# --- Write output ---
write.table(output_data, file = opt$output, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)

