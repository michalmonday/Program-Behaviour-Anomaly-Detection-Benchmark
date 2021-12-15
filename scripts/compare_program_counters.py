#!/usr/bin/python3
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import sys

def read_values(f):
    with f:
        return [int(line.strip(), 16) for line in f.readlines() if line]

 # pd.DataFrame(pc, columns=[fname])

if __name__ == '__main__':
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

    all_pc = []
    for f in args.pc_files:
        all_pc.append( read_values(f) )

    df = pd.DataFrame(all_pc, dtype=np.uint64, index=[f.name for f in args.pc_files]).T
    ax = df.plot.hist(bins=100, alpha=1/df.shape[1])
    ax.get_xaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')
    ax.get_yaxis().set_major_formatter(lambda x,pos: f'{int(x)}')
    plt.show()

    
    import pdb; pdb.set_trace()
    
    
    

