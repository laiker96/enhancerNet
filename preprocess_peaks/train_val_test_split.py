import pybedtools as pybed
import os
import argparse
from typing import Tuple

def train_val_test_split(bed_file: str, 
                         val_interval: str, 
                         test_interval: str) -> Tuple[str, str, str]:
    
    '''
    Split a BED file into training, validation, and test sets based on chromosome intervals.

    Parameters:
    - bed_file (str): Path to the input BED file.
    - val_interval (str): Chromosome interval for the validation set in the format "chromosome:start-end".
    - test_interval (str): Chromosome interval for the test set in the format "chromosome:start-end".

    Returns:
    - Tuple of str: Paths to the test, validation, and training BED files.
    '''
    
    output_prefix = os.path.splitext(bed_file)[0]
    
    bed_file = pybed.BedTool(bed_file)
    
    val_chrom, val_range = val_interval.split(':')
    val_range = [int(i) for i in val_range.split('-')]
    
    test_chrom, test_range = test_interval.split(':')
    test_range = [int(i) for i in test_range.split('-')]
    
    test_data = bed_file.filter(lambda x: x.chrom == test_chrom)
    test_data = test_data.filter(lambda x: x.start >= test_range[0] and x.end <= test_range[1])
    
    val_data = bed_file.filter(lambda x: x.chrom == val_chrom)
    val_data = val_data.filter(lambda x: x.start >= val_range[0] and x.end <= val_range[1])
    
    train_data = bed_file.filter(lambda x: not(x.chrom in [val_chrom, test_chrom]))
    
    test_data.saveas(output_prefix + '_test.bed')
    val_data.saveas(output_prefix + '_val.bed')
    train_data.saveas(output_prefix + '_train.bed')
    return (output_prefix + '_test.bed', output_prefix + '_val.bed', output_prefix + '_train.bed')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process BED file and split it into train, validation, and test sets.')
    parser.add_argument('--bed_filename', type = str, 
                        help = 'Path to the BED file input.')
    parser.add_argument('--val_interval', type = str, 
                        help = 'Validation set interval in the format "chromosome:start-end". Default is "none".', default = 'none')
    parser.add_argument('--test_interval', type = str, 
                        help = 'Test set interval in the format "chromosome:start-end". Default is "none".', default = 'none')

    args = parser.parse_args()
    args_list = list(vars(args).values())
    train_val_test_split(*args_list)

