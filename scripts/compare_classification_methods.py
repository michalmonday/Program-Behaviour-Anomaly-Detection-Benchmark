#!/usr/bin/python3.7

'''
# Example run:
./% --normal-pc ../log_files/stack-mission_riscv64_normal.pc --abnormal-pc ../log_files/stack-mission_riscv64_compromised.pc --function-ranges ../log_files/stack-missi on_riscv64_llvm_objdump_ranges.json

./% -n ../log_files/*normal*pc -a ../log_files/*compr*pc -fr ../log_files/*json --window-size 40 --epochs 30 --abnormal-load-address 4 --relative-pc --transition-sequence-size 3 --ignore-non-jumps --autoencoder-forest-size 30
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
        required=False,
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


# parser.add_argument(
#         '--relative-pc',
#         required=False,
#         action='store_true',
#         help='Converts absolute program counter values into relative ones.'
#         )
# 
# parser.add_argument(
#         '--ignore-non-jumps',
#         required=False,
#         action='store_true',
#         help='Ignores PC counter values that were likely not to be result of a jump (if previous counter value was not different by more than 4, then value will be ignored, in both: training and testing data)'
#         )
# 
# parser.add_argument(
#         '--window-size',
#         required=False,
#         type=int,
#         default=20,
#         metavar='',
#         help='Window size used for LSTM autoencoder method (+ possibly others in future'
#         )
# 
# parser.add_argument(
#         '--epochs',
#         required=False,
#         type=int,
#         default=10,
#         metavar='',
#         help='For how many epochs to train LSTM autoencoder method (+ possibly others in future'
#         )
# 
# parser.add_argument(
#         '--abnormal-load-address',
#         required=False,
#         type=int,
#         default=0,
#         metavar='',
#         help=('Specified value will be added to every program counter'
#               ' (to pretend that the program was loaded at different offset,'
#               ' making the detection more difficult and realistic).')
#         )
# 
# parser.add_argument(
#         '--transition-sequence-size',
#         required=False,
#         type=int,
#         default=2,
#         metavar='',
#         help='How many program counter transitions to take into account for "unique transitions" method (2 by default)'
#         )
# 
# parser.add_argument(
#         '--autoencoder-forest-size',
#         required=False,
#         type=int,
#         default=1,
#         metavar='',
#         help='How many autoencoder models to use that focus on its own standard deviation range.'
#         )
# 
# parser.add_argument(
#         '--introduce-artificial-anomalies',
#         required=False,
#         action='store_true',
#         help='Modifies the abnormal files (using utils.introduce_artificial_anomalies function)'
#         )

args = parser.parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import json

import utils
from utils import read_pc_values, plot_pc_histogram, plot_pc_timeline, df_from_pc_files, plot_vspans, plot_vspans_ranges, print_config
from utils import Artificial_Anomalies
from lstm_autoencoder import lstm_autoencoder
from unique_transitions import unique_transitions

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

import configparser
conf = configparser.ConfigParser()
conf.read('compare_classification_methods_config.ini')
logging.info(f'Loaded config from: compare_classification_methods_config.ini')
print_config(conf)


if __name__ == '__main__':

    #########################################
    # Load and plot data

    # Load function_ranges (used just for plotting)
    function_ranges = json.load(args.function_ranges) if args.function_ranges else {}
    # Load program counter values from files
    df_n = df_from_pc_files(
            args.normal_pc, 
            column_prefix    = 'normal: ',
            relative_pc      = conf['data'].getboolean('relative_pc'),
            ignore_non_jumps = conf['data'].getboolean('ignore_non_jumps')
            )

    anomalies_ranges = []
    pre_anomaly_values = []
    if args.abnormal_pc:
        df_a = df_from_pc_files(
                args.abnormal_pc,
                column_prefix    = 'abnormal: ',
                relative_pc      = conf['data'].getboolean('relative_pc'),
                ignore_non_jumps = conf['data'].getboolean('ignore_non_jumps'),
                load_address     = conf['data'].getint('abnormal_load_address')
                )
    else:
        df_a = pd.DataFrame()
        all_anomaly_methods = [
                Artificial_Anomalies.randomize_section,
                Artificial_Anomalies.slightly_randomize_section,
                Artificial_Anomalies.minimal
                ]
        # Introduce artificial anomalies
        for method in all_anomaly_methods:
                # prev_df_a = df_a.copy()
                # df_a, anomalies_ranges, pre_anomaly_values = utils.introduce_artificial_anomalies(df_a)
                # import pdb; pdb.set_trace()
            for i, column_name in enumerate(df_n):
                col_a, anomalies_ranges, pre_anomaly_values = method(df_n[column_name].copy())
                column_name = column_name.replace('normal', f'abnormal_{i}_{method.__name__}')
                df_a[column_name] = col_a

    logging.info(f'Number of normal pc files: {df_n.shape[1]}')
    logging.info(f'Number of abnormal pc files: {df_a.shape[1]}')

    # if conf['data'].getboolean('introduce_artificial_anomalies'):

    # Plot training (normal pc) and testing (abnormal/compromised pc) data
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace=0.43, top=0.835)
    fig.suptitle('TRAINING AND TESTING DATA', fontsize=20)
    ax  = plot_pc_timeline(df_n, function_ranges, ax=axs[0], title='Normal program counters - used for training')
    ax2 = plot_pc_timeline(df_a, function_ranges, ax=axs[1], title='Abnormal program counters - used for testing')
    # Plot artificial anomalies
    if conf['data'].getboolean('introduce_artificial_anomalies'):
        plot_vspans_ranges(ax2, anomalies_ranges, color='blue', alpha=0.05)
        for vals in pre_anomaly_values:
            vals.plot(ax=ax2, color='purple', marker='h', markersize=2, linewidth=0.7, linestyle='dashed')
            ax2.fill_between(vals.index.values, vals.values, df_a.loc[vals.index].values.reshape(-1), color='r', alpha=0.15)


    #########################################
    # Unique transitions

    n = conf['unique_transitions'].getint('sequence_size')

    unique_transitions.train(df_n, n=n)
    detected_ut, df_a_detected_points = unique_transitions.predict(df_a)

    ax3 = plot_pc_timeline(df_a, function_ranges, title=f'UNIQUE TRANSITIONS METHOD (n={n}) RESULTS')
    df_a_detected_points.plot(ax=ax3, color='r', marker='*', markersize=10, linestyle='none', legend=None)
    plot_vspans(ax3, df_a_detected_points.index.values - n+1, n-1, color='red')


    #########################################
    # LSTM autoencoder

    window_size = conf['lstm_autoencoder'].getint('window_size')
    lstm_autoencoder.train(df_n, window_size=window_size, epochs=conf['lstm_autoencoder'].getint('epochs'), number_of_models=conf['lstm_autoencoder'].getint('forest_size'))
    results_df, anomalies_df = lstm_autoencoder.predict(df_a)
    axs = lstm_autoencoder.plot_results(df_a, results_df, anomalies_df, window_size, fig_title = 'LSTM AUTOENCODER RESULTS', function_ranges=function_ranges)

    plt.show()
    

