#!/bin/bash

# Usage: ./augment_bed.sh input.bed output.bed

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input_bed> <output_bed>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

{
  awk 'BEGIN {FS=OFS="\t"} {
    printf "%s\t1000\t+\t", $1 FS $2 FS $3 FS $4
    for (i = 5; i <= NF; i++) {
      printf "%s", $i
      if (i < NF) printf OFS; else printf "\n"
    }
  }' "$INPUT"

  awk 'BEGIN {FS=OFS="\t"} {
    printf "%s\t1000\t-\t", $1 FS $2 FS $3 FS $4
    for (i = 5; i <= NF; i++) {
      printf "%s", $i
      if (i < NF) printf OFS; else printf "\n"
    }
  }' "$INPUT"
} | sort-bed --max-mem 32G - > "$OUTPUT"
