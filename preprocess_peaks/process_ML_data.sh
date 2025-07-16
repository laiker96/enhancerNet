bedtools makewindows -i srcwinnum -w 1000 -s 50 -g genome_files/dm6.chrom.sizes | awk '$3-$2 == 1000' | awk '$1 == "chr2R" || $1 == "chr2L" || $1 == "chr3R" || $1 == "chr3L" || $1 == "chr4" || $1 == "chrX"' | bedtools intersect -wa -v -a - -b genome_files/dm6-blacklist.v2.bed | sort-bed --max-mem 8G - > genome_files/windows_1kb_50bp.bed

cat peak_files/peaks_all_contexts/*/*.narrowPeak peak_files/H3K27ac/*.bed | awk 'BEGIN {FS="\t";OFS="\t"} {print $1,$2+$10,$2+$10+1,$4,$5,$6,$7,$8,$9,$10}' | sort-bed --max-mem 8G - | bedtools slop -g genome_files/dm6.chrom.sizes -r 150 -l 149 -i - | awk '$3-$2==300' > ML_datasets/S3_norm_unfiltered/summits_extended_300bp_new.bed

cp genome_files/windows_1kb_50bp.bed reference.bed

while read SRA; do                                                                                                                                                                                   
bigWigAverageOverBed signal_files/unfiltered/standard_all_contexts/S3norm_rc_bedgraph/"$SRA"_S3.bw genome_files/windows_1kb_50bp.bed -bedOut=tmp.bed -sampleAroundCenter=300 out.tab && cut -f 5 tmp.bed | paste reference.bed - > tmp && mv tmp reference.bed ; rm tmp.bed out.tab
done < <(tail -n +2 metadata/contexts_complete_data.csv | cut -f 5 -d ,)

awk 'BEGIN {FS="\t";OFS="\t"} {print $1,$2 + int(($3-$2)/2)-150,$2 + int(($3-$2)/2)+150,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13}' reference.bed > central_300bp_reference.bed

bedmap --echo --delim '\t' --bases-uniq-f central_300bp_reference.bed ML_datasets/S3_norm_unfiltered/summits_extended_300bp_new.bed | cut -f 1-4 --complement - | paste genome_files/windows_1kb_50bp.bed - > ML_datasets/S3_norm_unfiltered/indicator_values_1kb_300bp.bed
