#!/usr/bin/python3.7

'''
# Example run:
./% --normal-pc ../log_files/stack-mission_riscv64_normal.pc --abnormal-pc ../log_files/stack-mission_riscv64_compromised.pc --function-ranges ../log_files/stack-missi on_riscv64_llvm_objdump_ranges.json

./% -n ../log_files/*normal*pc -a ../log_files/*compr*pc -fr ../log_files/*json --relative-pc --window-size 20 --epochs 50
'''


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

parser.add_argument(
        '--relative-pc',
        required=False,
        action='store_true',
        help='Converts absolute program counter values into relative ones.'
        )

parser.add_argument(
        '--ignore-non-jumps',
        required=False,
        action='store_true',
        help='Ignores PC counter values that were likely not to be result of a jump (if previous counter value was not different by more than 4, then value will be ignored, in both: training and testing data)'
        )

parser.add_argument(
        '--window-size',
        required=False,
        type=int,
        default=20,
        metavar='',
        help='Window size used for LSTM autoencoder method (+ possibly others in future'
        )

parser.add_argument(
        '--epochs',
        required=False,
        type=int,
        default=10,
        metavar='',
        help='For how many epochs to train LSTM autoencoder method (+ possibly others in future'
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
from lstm_autoencoder import lstm_autoencoder
from unique_transitions import unique_transitions

# ax = plot_pc_histogram(df, function_ranges, bins=100)
# ax2 = plot_pc_timeline(df, function_ranges)
# df = df_from_pc_files(f_list)
# pc = read_pc_values(f)


if __name__ == '__main__':
    # function_ranges are used just for plotting
    function_ranges = json.load(args.function_ranges) if args.function_ranges else {}
    df_n = df_from_pc_files(args.normal_pc, column_prefix='normal: ', relative_pc=args.relative_pc, ignore_non_jumps=args.ignore_non_jumps)
    df_a = df_from_pc_files(args.abnormal_pc, column_prefix='abnormal: ', relative_pc=args.relative_pc, ignore_non_jumps=args.ignore_non_jumps)

    # plot training (normal pc) and testing (abnormal/compromised pc) data
    fig, axs = plt.subplots(2)
    # fig.subplots_adjust(top=0.92, hspace=0.43)
    fig.subplots_adjust(hspace=0.43, top=0.835)
    fig.suptitle('TRAINING AND TESTING DATA', fontsize=20)
    ax  = plot_pc_timeline(df_n, function_ranges, ax=axs[0], title='Normal program counters - used for training')
    ax2 = plot_pc_timeline(df_a, function_ranges, ax=axs[1], title='Abnormal program counters - used for testing')

    detected_ut, df_a_detected_points = unique_transitions.detect(df_n, df_a)
    # df_a_detected_points.plot(ax=ax2, color='r', marker='*', markersize=10, linestyle='none', legend=None)

    ax3 = plot_pc_timeline(df_a, function_ranges, title='UNIQUE TRANSITIONS METHOD RESULTS')
    df_a_detected_points.plot(ax=ax3, color='r', marker='*', markersize=10, linestyle='none', legend=None)

    # LSTM autoencoder
    results_df, anomalies_df = lstm_autoencoder.detect(df_n, df_a, window_size=args.window_size, epochs=args.epochs)
    lstm_autoencoder.plot_results(df_a, results_df, anomalies_df, args.window_size, fig_title = 'LSTM AUTOENCODER RESULTS', function_ranges=function_ranges)

    plt.show()
    

