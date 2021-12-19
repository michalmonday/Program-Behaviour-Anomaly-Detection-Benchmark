#!/usr/bin/python3

# Example run:

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import sys
import json

def read_values(f):
    with f:
        return [int(line.strip(), 16) for line in f.readlines() if line]

def plot(df, function_ranges={}, bins=100):
    ax = df.plot.hist(bins=bins, alpha=1/df.shape[1])
    ax.get_xaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')
    ax.get_yaxis().set_major_formatter(lambda x,pos: f'{int(x)}')
    # import pdb; pdb.set_trace()
    x_start = ax.get_xticks()[0]
    x_end = ax.get_xticks()[-1]
    y_top = ax.transAxes.to_values()[3]
    i = 0
    function_line_width = 0.7
    for func_name, (start, end) in function_ranges.items():
        if func_name == 'total': continue
        if start < x_start or end > x_end: continue
        ax.axvline(x=start, color='black', linewidth=function_line_width)
        # ax.axvline(x=end, color='black')
        # ax.text(start, y_top*(0.8-i/100), func_name, rotation=90, va='bottom', fontsize='x-small')
        # ax.text(start, y_top*0.7, func_name, rotation=90, va='bottom', fontsize='x-small')
        i += 1
    # ax_t.xaxis.tick_top()
    # ax.legend(bbox_to_anchor=(0.1, 0.7))
    ax_t = ax.twiny()
    ax_t.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
    ax_t.set_xticks([v[0] for v in function_ranges.values()])
    ax_t.set_xticklabels(list(function_ranges.keys()), fontdict={'fontsize':7}, rotation=90)
    ax_t.set_xlim(*ax.get_xlim())

    ax2 = df.plot(linewidth=0.7)
    ax2.get_yaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')
    i = 0
    for func_name, (start, end) in function_ranges.items():
        if func_name == 'total': continue
        if start < x_start or end > x_end: continue
        ax2.axhline(y=start, color='black', linewidth=function_line_width)
        # ax2.text(df.shape[0]*(0.9-i/80), start, func_name, rotation=0, va='bottom', fontsize='x-small')
        # ax2.text(df.shape[0]*0.8, start, func_name, rotation=0, va='bottom', fontsize='x-small')
        i += 1
    ax2_t = ax2.twinx()
    ax2_t.set_yticks([v[0] for v in function_ranges.values()])
    ax2_t.set_yticklabels(list(function_ranges.keys()), fontdict={'fontsize':7})
    ax2_t.set_ylim(*ax2.get_ylim())

    df.plot(ax=ax2, marker='h', markersize=1, linestyle='none', legend=None)
    plt.show()

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

    if args.function_ranges:
        plot(df, json.load(args.function_ranges) )
    else:
        plot(df)
    
    import pdb; pdb.set_trace()
    
    
    

