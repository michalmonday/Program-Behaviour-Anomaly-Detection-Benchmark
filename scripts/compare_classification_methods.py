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

parser.add_argument(
        '--plot-data',
        required=False,
        action='store_true',
        help='Plots training and testing data'
        )

parser.add_argument(
        '--quick-test',
        required=False,
        action='store_true',
        help='Overrides sequence/window sizes for quick testing.'
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
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import json
from copy import deepcopy

import utils
from utils import read_pc_values, plot_pc_histogram, plot_pc_timeline, df_from_pc_files, plot_vspans, plot_vspans_ranges, print_config
from artificial_anomalies import Artificial_Anomalies
from lstm_autoencoder import lstm_autoencoder
from lstm_autoencoder.lstm_autoencoder import LSTM_Autoencoder
from unique_transitions.unique_transitions import Unique_Transitions
from detection_model import Detection_Model

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

import configparser
conf = configparser.ConfigParser()
conf.read('compare_classification_methods_config.ini')
logging.info(f'Loaded config from: compare_classification_methods_config.ini')
print_config(conf)


def plot_data(df_n, df_a, function_ranges={}, anomalies_ranges=[], pre_anomaly_values=[]):
    # Plot training (normal pc) and testing (abnormal/compromised pc) data
    ax = plot_pc_timeline(df_n, function_ranges)
    ax.set_title('TRAIN DATA', fontsize=20)
    fig, axs = plt.subplots(df_a.shape[1], sharex=True)#, sharey=True)
    # fig.subplots_adjust(hspace=0.43, top=0.835)
    fig.subplots_adjust(hspace=1.4, top=0.835)
    fig.suptitle('TEST DATA', fontsize=20)
    # fig.supxlabel('Instruction index')
    # fig.supylabel('Program counter (address)')
    fig.text(0.5, 0.04, 'Instruction index', ha='center')
    fig.text(0.04, 0.5, 'Program counter (address)', va='center', rotation='vertical')

    for i, col_name in enumerate(df_a):
        ax = axs[i]
        plot_pc_timeline(df_a[col_name], function_ranges, ax=ax, title=col_name, xlabel='', ylabel='')
        # Plot artificial anomalies
        if not args.abnormal_pc:
            plot_vspans_ranges(ax, anomalies_ranges[i], color='blue', alpha=0.05)
            for vals in pre_anomaly_values[i]:
                vals.plot(ax=ax, color='purple', marker='h', markersize=2, linewidth=0.7, linestyle='dashed')
                # ax.fill_between(vals.index.values, vals.values, df_a.loc[vals.index].values.reshape(-1), color='r', alpha=0.15)
                ax.fill_between(vals.index.values, vals.values, df_a.loc[vals.index][col_name].values.reshape(-1), color='r', alpha=0.15)


if __name__ == '__main__':

    if args.quick_test:
        logging.info('\nOVERRIDING CONFIG WITH VALUES FOR QUICK TESTING (because --quick-test was supplied)')
        logging.info('Overriden values are: sequence/window sizes, forest size, epochs.\n')
        conf['unique_transitions']['sequence_sizes'] = '2'
        conf['lstm_autoencoder']['window_sizes'] = '3'
        conf['lstm_autoencoder']['forest_size'] = '3'
        conf['lstm_autoencoder']['epochs'] = '5'
        conf['data']['artificial_anomalies_offsets_count'] = '5'

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
        df_a_ground_truth = pd.DataFrame(dtype=bool)
        all_anomaly_methods = [
                Artificial_Anomalies.randomize_section,
                Artificial_Anomalies.slightly_randomize_section,
                Artificial_Anomalies.minimal
                ]
        # Introduce artificial anomalies for all the files, resulting in the following testing examples:
        # - method 1 with file 1
        # - method 1 with file 2
        # - method 2 with file 1
        # - method 2 with file 2
        # - method 3 with file 1
        # - method 3 with file 2
        # 
        # Where "method" is a one of the methods from "Artificial_Anomalies" class (e.g. randomize_section, slightly_randomize_section, minimal)
        # and where "file" is a normal/baseline file containing program counters.
        # Example above shows only 3 methods and 2 files, but the principle applies for any number.
        # So with 5 methods and 5 normal pc files there would be 25 testing examples.

        offsets_count = conf['data'].getint('artificial_anomalies_offsets_count')
        for i, method in enumerate(all_anomaly_methods):

            # for each normal/baseline append column with introduced anomalies into into "df_a"
            for j, column_name in enumerate(df_n):
                for k in range(offsets_count):
                    # introduce anomalies
                    col_a, ar, pav, col_a_ground_truth = method(df_n[column_name].copy())
                    # keep record of anomalies and previous values (for plotting later)
                    anomalies_ranges.append(ar)
                    pre_anomaly_values.append(pav)
                    # rename column
                    new_column_name = column_name.replace('normal', f'{method.__name__}_({i},{j},{k})', 1)
                    df_a[new_column_name] = col_a
                    df_a_ground_truth[new_column_name] = col_a_ground_truth

    logging.info(f'Number of normal pc files: {df_n.shape[1]}')
    logging.info(f'Number of abnormal pc files: {df_a.shape[1]}')

    if args.plot_data:
        plot_data(
                df_n,
                df_a,
                function_ranges=function_ranges,
                anomalies_ranges=anomalies_ranges,
                pre_anomaly_values=pre_anomaly_values
                )

    df_results = pd.DataFrame(columns=Detection_Model.evaluation_metrics)

    if conf['unique_transitions'].getboolean('active'):
        # Unique transitions
        sequence_sizes = [int(seq_size) for seq_size in conf['unique_transitions'].get('sequence_sizes').strip().split(',')]
        for seq_size in sequence_sizes:
            ut = Unique_Transitions()
            ut.train(df_n, n=seq_size)

            # results_ua is a list of boolean lists for each file
            # where True=anomaly, False=normal
            results_ut = ut.predict_all(df_a)
            df_a_ground_truth_windowized = utils.windowize_ground_truth_labels(
                    df_a_ground_truth,
                    seq_size # window/sequence size
                    )

            # anomaly_recall = what percent of anomalies will get detected
            # false_positives_ratio = what percent of normal program behaviour 
            #                         will be classified as anomalous, which is 
            #                         referred to as "false positives" in other
            #                         papers (about anomaly detection)
            em = ut.evaluate_all(results_ut, df_a_ground_truth_windowized)
            logging.info( ut.format_evaluation_metrics(em) )

            method_name = f'unique_transitions (seq_size={seq_size})'
            df_results.loc[method_name] = em

    if conf['lstm_autoencoder'].getboolean('active'):
        # LSTM autoencoder
        # window_size = conf['lstm_autoencoder'].getint('window_size')
        window_sizes = [int(ws) for ws in conf['lstm_autoencoder'].get('window_sizes').strip().split(',')]
        for window_size in window_sizes:
            la = LSTM_Autoencoder()
            la.train(df_n, window_size=window_size, epochs=conf['lstm_autoencoder'].getint('epochs'), number_of_models=conf['lstm_autoencoder'].getint('forest_size'))

            # results_lstm is a list of tuples where each tuple has:
            # - is_anomaly (bool)
            # - results_df (df with columns: loss, threshold, anomaly, window_start, window_end)
            # - anomalies_df (just like results_df but only containing rows for anomalous windows)
            results_lstm = la.predict_all(df_a)
            df_a_ground_truth_windowized = utils.windowize_ground_truth_labels(
                    df_a_ground_truth,
                    window_size 
                    )
            em = la.evaluate_all(results_lstm, df_a_ground_truth_windowized)
            logging.info( la.format_evaluation_metrics(em) )
            # logging.info(f'LSTM autoencoder accuracy: {accuracy_lstm:.2f}')
            # logging.info(f'LSTM autoencoder false positives: {false_positives_lstm:.2f}')
            method_name = f'lstm autoencoder (window_size={window_size})'
            df_results.loc[method_name] = em


    axs = df_results[['anomaly_recall', 'false_positives_ratio']].plot.bar(rot=15, subplots=True)
    for ax in axs:
        for container in ax.containers:
            # set numerical label on top of bar/rectangle
            ax.bar_label(container)
    plt.show()

    #for col_a in df_a:
    #    #########################################
    #    # Test models

    #    # Unique transitions
    #    is_anomalous, detected_ut, df_a_detected_points = ut.predict(df_a[col_a])
    #    # ax3 = plot_pc_timeline(df_a, function_ranges, title=f'UNIQUE TRANSITIONS METHOD (n={n}) RESULTS')
    #    # df_a_detected_points.plot(ax=ax3, color='r', marker='*', markersize=10, linestyle='none', legend=None)
    #    # plot_vspans(ax3, df_a_detected_points.index.values - n+1, n-1, color='red')

    #    # LSTM autoencoder
    #    is_anomalous, results_df, anomalies_df = la.predict(df_a[col_a])
    #    # axs = la.plot_results(df_a, results_df, anomalies_df, window_size, fig_title = 'LSTM AUTOENCODER RESULTS', function_ranges=function_ranges)

    #    # plt.show()
    

