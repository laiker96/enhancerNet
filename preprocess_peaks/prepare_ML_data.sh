#!/bin/bash

set -euo pipefail

# ------------------ HELP & ARGUMENT PARSING ------------------ #
usage() {
    echo "Usage: $0 -c chrom_sizes -b blacklist -p peak_dir -s signal_dir -d cre_dir -m metadata -o output_dir"
    exit 1
}

while getopts ":c:b:p:s:d:m:o:" opt; do
  case $opt in
    c) chrom_sizes="$OPTARG" ;;
    b) blacklist="$OPTARG" ;;
    p) peak_dir="$OPTARG" ;;
    s) signal_dir="$OPTARG" ;;
    d) cre_dir="$OPTARG" ;;
    m) metadata="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    *) usage ;;
  esac
done

if [ -z "${chrom_sizes:-}" ]; then echo "Missing: chrom_sizes"; usage; fi
if [ -z "${blacklist:-}" ]; then echo "Missing: blacklist"; usage; fi
if [ -z "${peak_dir:-}" ]; then echo "Missing: peak_dir"; usage; fi
if [ -z "${signal_dir:-}" ]; then echo "Missing: signal_dir"; usage; fi
if [ -z "${cre_dir:-}" ]; then echo "Missing: cre_dir"; usage; fi
if [ -z "${metadata:-}" ]; then echo "Missing: metadata"; usage; fi
if [ -z "${output_dir:-}" ]; then echo "Missing: output_dir"; usage; fi

mkdir -p "$output_dir"

# ------------------ FUNCTIONS ------------------ #

generate_windows() {
    local output="$1"
    bedtools makewindows -i srcwinnum -w 1000 -s 50 -g "$chrom_sizes" \
    | awk '$3 - $2 == 1000' \
    | awk '$1 ~ /^chr(2[LR]|3[LR]|4|X)$/' \
    | bedtools intersect -wa -v -a - -b "$blacklist" \
    | sort-bed --max-mem 8G - > "$output"
}

create_extended_summits() {
    local output="$1"
    cat "$peak_dir"/*/*.narrowPeak "$peak_dir"/H3K27ac/*.bed \
    | awk 'BEGIN {FS=OFS="\t"} {summit=$2+$(NF); print $1, summit, summit+1, $4, $5, $6, $7, $8, $9, $(NF)}' \
    | sort-bed --max-mem 8G - \
    | bedtools slop -g "$chrom_sizes" -r 150 -l 149 -i - \
    | awk '$3 - $2 == 300' > "$output"
}

create_map_cres() {
    local output="$1"
    cat "$peak_dir"/*/*.narrowPeak "$peak_dir"/H3K27ac/*.bed "$cre_dir"/cCREs.bed \
    | awk 'BEGIN {FS=OFS="\t"} {print $1, $2, $3}' \
    | sort-bed --max-mem 8G - \
    | bedtools merge > "$output"
}

add_signal_values() {
    local window_bed="$1"
    local metadata_file="$2"
    local ref_output="$3"

    cp "$window_bed" "$ref_output"

    tail -n +2 "$metadata_file" | cut -f 5 -d , | while read -r SRA; do
        local bw_file="$signal_dir/${SRA}_S3.bw"
        bigWigAverageOverBed "$bw_file" "$window_bed" -bedOut=tmp.bed -sampleAroundCenter=300 out.tab
        cut -f 5 tmp.bed | paste "$ref_output" - > tmp && mv tmp "$ref_output"
        rm tmp.bed out.tab
    done
}

extract_centered_regions() {
    local input_ref="$1"
    local output="$2"
    local size="$3"  # total length of the centered region

    local half_size=$((size / 2))

    awk -v half="$half_size" 'BEGIN {FS=OFS="\t"} {
        mid = int(($2 + $3) / 2);
        start = mid - half;
        end = mid + half;
        printf "%s\t%d\t%d", $1, start, end;
        for (i=4; i<=NF; i++) printf "\t%s", $i;
        print ""
    }' "$input_ref" > "$output"
}

generate_labels() {
    local center_bed="$1"
    local summit_bed="$2"
    local window_bed="$3"
    local output="$4"

    bedmap --echo --delim '\t' --bases-uniq-f "$center_bed" "$summit_bed" \
    | cut -f 1-4 --complement - \
    | paste "$window_bed" - > "$output"
}

generate_labels_CREs() {
    local center_bed="$1"
    local map_regions_cCREs="$2"
    local window_bed="$3"
    local output="$4"
    local metadata_file="$5"
    local cre_dir="$6"
    local output_dir
    output_dir="$(dirname "$output")"
    local tmp_file="$output_dir/tmp"

    cp "$center_bed" "$output"

    tail -n +2 "$metadata_file" | cut -f 1 -d , | while read -r ID; do
        echo "Adding CRE label: dELS_${ID}"
        bedmap \
            --echo \
            --delim '\t' \
            --fraction-ref 0.5 \
            --indicator \
            "$output" \
            "$cre_dir/dELS_${ID}.bed" \
            > "$tmp_file" && mv "$tmp_file" "$output"
    done

    echo "Adding final cCRE map region label..."
    bedmap \
        --echo \
        --delim '\t' \
        --indicator \
        "$output" \
        "$map_regions_cCREs" \
        > "$tmp_file" && mv "$tmp_file" "$output"

    echo "Appending labels to window coordinates..."
    cut -f 1-4 --complement "$output" \
        | paste "$window_bed" - \
        > "$tmp_file" && mv "$tmp_file" "$output"
}

# ------------------ PIPELINE EXECUTION ------------------ #

window_bed="$output_dir/windows_1kb_50bp.bed"
summits_bed="$output_dir/summits_extended_300bp.bed"
map_cres="$output_dir/map_cres.bed"
reference_bed="$output_dir/reference.bed"
center_bed="$output_dir/central_300bp_reference.bed"
center_bed_500="$output_dir/central_500bp_reference.bed"
final_output="$output_dir/indicator_values_1kb_300bp.bed"
final_output_dELSs="$output_dir/indicator_values_1kb_dELSs.bed"

echo "Creating 1kb sliding windows..."
generate_windows "$window_bed"

echo "Extending summits to 300bp..."
create_extended_summits "$summits_bed"

echo "Create map cres..."
create_map_cres "$map_cres"

echo "Extracting signal values..."
add_signal_values "$window_bed" "$metadata" "$reference_bed"

echo "Extracting 300bp centers from 1kb windows..."
extract_centered_regions "$reference_bed" "$center_bed" 300

echo "Extracting 500bp centers from 1kb windows..."
extract_centered_regions "$window_bed" "$center_bed_500" 500

echo "Generating indicator labels..."
generate_labels "$center_bed" "$summits_bed" "$window_bed" "$final_output"

echo "Generating indicator labels for dELSs..."
generate_labels_CREs \
  "$center_bed_500" \
  "$map_cres" \
  "$window_bed" \
  "$final_output_dELSs" \
  "$metadata" \
  "$cre_dir"

echo "✅ Done! Output: $final_output"

