#!/usr/bin/python3

# Example run:
# !./% --normal-pc ../log_files/stack-mission_riscv64_normal.pc --abnormal-pc ../log_files/stack-mission_riscv64_compromised.pc --function-ranges ../log_files/stack-missi on_riscv64_llvm_objdump_ranges.json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        '-n',
        '--normal-pc',
        nargs='+',
        required=True,
        metavar='',
        type=argparse.FileType('r'),
        help='Program counter files (.pc) of "known/normal" programs as outputted by parse_qtrace_log.py'
        )

parser.add_argument(
        '-a',
        '--abnormal-pc',
        nargs='+',
        required=True,
        metavar='',
        type=argparse.FileType('r'),
        help='Program counter files (.pc) of "unknown/abnormal" programs as outputted by parse_qtrace_log.py'
        )

parser.add_argument(
        '-fr',
        '--function-ranges',
        type=argparse.FileType('r'),
        metavar='',
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

from utils import read_pc_values, plot_pc_histogram, plot_pc_timeline, df_from_pc_files
# ax = plot_pc_histogram(df, function_ranges, bins=100)
# ax2 = plot_pc_timeline(df, function_ranges)
# df = df_from_pc_files(f_list)
# pc = read_pc_values(f)


def unique_transitions(df):
    ''' returns unique transitions between program counters 
        Let's imagine that program consists of the following PC values:
            0, 4, 8, 4, 8, 4, 8, 12
    
        In that case, the returned unique transitions would be:
            (0,4)
            (4,8)
            (8,4)
            (8,12)

        The returned unique transitions may be used as a Finite State 
        Automaton during program execution. Any observed transition will
        indicate abnormal behaviour. This approach was used in:
            "2001 - A Fast Automaton-Based Method for Detecting Anomalous Program Behaviors by Sekar et al."

        This method is simple and effective, however it works on assumption that
        normal program goes into every possible valid state during training.
        In embedded systems where program execution is determined by the intricate
        state of sensors or user input, this isn't possible.

        How to extract unique transitions from pandas dataframe containing PC counters?
        - stack all program runs PC values (multiple columns) on top of each other (separated by "NaN") in a single column
        - clone all_pc column and shift the cloned version by -1
        - drop rows with NaN
        - get unique pairs from 2 columns
    '''
    # add NaN row to each column (to avoid recognizing the last PC of 1 run as first PC of 2nd run)
    df = df.append(pd.Series(), ignore_index=True)
    # stack all columns on top of each other
    df = df.melt(value_name='all_pc').drop('variable',1)
    # clone all_pc column and shift it by -1
    df['all_pc_shifted'] = df['all_pc'].shift(-1)
    # drop any rows with at least 1 NaN value ( as seen here: https://stackoverflow.com/a/13434501/4620679 )
    df = df.dropna()
    # get unique transitions
    unique_transitions = df.drop_duplicates()
    return unique_transitions

def detect_by_unique_transitions(df_n, df_a):
    normal_ut = unique_transitions(df_n)
    abnormal_ut = unique_transitions(df_a)
    # gets abnormal_ut entries that are not present in normal_ut
    # (it ignores df index so that's why it's ugly)
    # it's from: https://stackoverflow.com/a/50645672/4620679
    detected_ut = abnormal_ut[ ~abnormal_ut[ ~abnormal_ut.stack().isin(normal_ut.stack().values).unstack()].isna().all(axis=1) ].dropna()

    # At this point, if detected_ut is not empty, it means that abnormal behaviour was detected.
    # We can further plot where exactly it happened during program execution.

    # get PC values in abnormal run where unseen transitions are observed
    df_a_detected_points = df_a[ df_a[df_a.columns[0]].rolling(2).apply(lambda x: ((detected_ut['all_pc'] == x.iloc[0]) & (detected_ut['all_pc_shifted'] == x.iloc[1])).any() ) > 0.0 ]
    return detected_ut, df_a_detected_points


if __name__ == '__main__':
    # function_ranges are used just for plotting
    function_ranges = json.load(args.function_ranges) if args.function_ranges else {}
    df_n = df_from_pc_files(args.normal_pc, column_prefix='normal: ')
    df_a = df_from_pc_files(args.abnormal_pc, column_prefix='abnormal: ')
    ax  = plot_pc_timeline(df_n, function_ranges)
    ax2 = plot_pc_timeline(df_a, function_ranges)

    detected_ut, df_a_detected_points = detect_by_unique_transitions(df_n, df_a)
    df_a_detected_points.plot(ax=ax2, color='r', marker='*', markersize=10, linestyle='none', legend=None)

    plt.show()
    
    import pdb; pdb.set_trace()
    
    
    

