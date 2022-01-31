#!/usr/bin/python3

# Example run:
# !./% ../log_files/stack-mission_riscv64_compromised.pc ../log_files/stack-mission_riscv64_normal.pc -fr ../log_files/stack-mission_riscv64_llvm_objdump_ranges.json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        'pc_files',
        nargs='+',
        type=argparse.FileType('r'),
        help='Program counter files (.pc) as outputted by parser_simple.py'
        )

parser_group = parser.add_mutually_exclusive_group(required=True)
parser_group.add_argument(
        '-fr',
        '--function-ranges',
        type=argparse.FileType('r'),
        help='File name containing output of extract_function_ranges_from_llvm_objdump.py'
        )

args = parser.parse_args()

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import json

from utils import read_pc_values, plot_pc_histogram, plot_pc_timeline


if __name__ == '__main__':
    all_pc = []
    for f in args.pc_files:
        all_pc.append( read_pc_values(f) )

    df = pd.DataFrame(all_pc, dtype=np.uint64, index=[f.name for f in args.pc_files]).T

    function_ranges = json.load(args.function_ranges) if args.function_ranges else {}

    ax = plot_pc_histogram(df, function_ranges, bins=100)
    ax2 = plot_pc_timeline(df, function_ranges)
    
    plt.show()
    import pdb; pdb.set_trace()
    
    
    

