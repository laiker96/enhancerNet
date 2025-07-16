library(tidyverse)
library(tibble)

ids <- read.delim('headers.txt', header=F)$V1
name_matches <- read.csv("../../metadata/contexts_complete_data.csv", header=T, sep = ",")
ids <- name_matches[match(x = ids, table = name_matches$SRA.ID), "context"]
ids <- c("chr", "start", "end", "name", ids)


data <- read.delim('test_dir/indicator_values_1kb_300bp.bed', header=F)
str(data)
data <- data[complete.cases(data),]
data_copy <- data

upper_limits <- sapply(X = data[,seq(5,13)], FUN = quantile, probs = 0.995)
lower_limits <- sapply(X = data[,seq(5,13)], FUN = quantile, probs = 0.1)
#lower_limits[4] <- quantile(data$V8, probs = 0.15)
#lower_limits[5] <- quantile(data$V9, probs = 0.15)
#lower_limits[6] <- quantile(data$V10, probs = 0.15)
#lower_limits[14] <- quantile(data$V18, probs = 0.15)


for (j in seq(1, length(upper_limits))) {
  
  data <- data[(data[, (4 + j)] < upper_limits[j]), ]
  data <- data[(data[, (4 + j)] > lower_limits[j]), ]
  
}


data[,seq(5,13)] <- log2(data[,seq(5,13)])

data %>% filter(V14 >= 0.7) -> positive_samples
data %>% filter(V14 < 0.7) -> negative_samples
#data %>% filter(V14 == 0) -> negative_samples

sampled_negatives <- negative_samples[sample(nrow(negative_samples), 
                                             as.integer(dim(positive_samples)[1])/4), ]


rbind(positive_samples, sampled_negatives)[,-c(14)] -> output_data
names(output_data) <- ids
hist(output_data$LB, breaks = 100)

boxplot(output_data[,-c(1:4)])

write.table(output_data, "dataset_1kb_300bp_S3.bed", quote = F, sep = '\t', 
            row.names = F, col.names=F)






