#!/usr/bin/python3.7

'''
! clear; python3.7 % -n ../../log_files/*normal*pc -a ../../log_files/*compr*pc -fr ../../log_files/*json
'''


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import json

import os
import sys
import inspect
from sklearn.metrics import precision_recall_fscore_support


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import logging

import utils
from utils import read_pc_values, plot_pc_histogram, plot_pc_timeline, df_from_pc_files

# ax = plot_pc_histogram(df, function_ranges, bins=100)
# ax2 = plot_pc_timeline(df, function_ranges)
# df = df_from_pc_files(f_list)
# pc = read_pc_values(f)

#def unique_transitions(df, n=2):
#    ''' returns unique transitions between program counters 
#        Let's imagine that program consists of the following PC values:
#            4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 12
#    
#        In that case, the returned unique transitions would be:
#            (4,8)
#            (8,4)
#            (8,12)
#
#        The returned unique transitions may be used as a Finite State 
#        Automaton during program execution. Any observed transition will
#        indicate abnormal behaviour. This approach was used in:
#            "2001 - A Fast Automaton-Based Method for Detecting Anomalous Program Behaviors by Sekar et al."
#
#        This method is simple and effective, however it works on assumption that
#        normal program goes into every possible valid state during training.
#        In embedded systems where program execution is determined by the intricate
#        state of sensors or user input, this isn't possible.
#
#        How to extract unique transitions from pandas dataframe containing PC counters?
#        - stack all program runs PC values (multiple columns) on top of each other (separated by "NaN") in a single column
#        - clone all_pc column and shift the cloned version by -1
#        - drop rows with NaN
#        - get unique pairs from 2 columns
#    '''
#    # add NaN row to each column (to avoid recognizing the last PC of 1 run as first PC of 2nd run)
#    # later "df.dropna()" will just remove these, fixing the problem arising from stacking
#    # program counters from multiple files on each other
#    df = df.append(pd.Series(), ignore_index=True)
#    # stack all columns on top of each other
#    df = df.melt(value_name='all_pc').drop('variable', axis=1)
#    # clone all_pc column and shift it by -1
#    df['all_pc_shifted'] = df['all_pc'].shift(-1)
#    # drop any rows with at least 1 NaN value ( as seen here: https://stackoverflow.com/a/13434501/4620679 )
#    df = df.dropna()
#    # get unique transitions
#    unique_transitions = df.drop_duplicates()
#    return unique_transitions

class Unique_Transitions:
    def __init__(self):
        self.normal_ut = None
        self.train_n = None

    def train(self, df_n, n=2):
        utils.print_header(f'UNIQUE TRANSITIONS (n={n})')
        self.normal_ut = utils.pc_df_to_sliding_windows(df_n, window_size=n, unique=True)
        self.train_n = n

        logging.info(f'Number of train programs: {df_n.shape[1]}')
        logging.info(f'Longest train program size: {df_n.shape[0]} instructions')
        logging.info(f'Number of unique train sequences (with size of {n}): {self.normal_ut.shape[0]}')

    def predict(self, df_a):
        # gets abnormal_ut entries that are not present in normal_ut
        # (it ignores df index so that's why it's ugly)
        # it's from: https://stackoverflow.com/a/50645672/4620679
        # import pdb; pdb.set_trace()

        col_a = pd.DataFrame(df_a)

        abnormal_ut = utils.pc_df_to_sliding_windows(col_a, window_size=self.train_n, unique=True)
        detected_ut = abnormal_ut[ ~abnormal_ut[ ~abnormal_ut.stack().isin(self.normal_ut.stack().values).unstack()].isna().all(axis=1) ].dropna()
       
        # set is used for fast lookup
        detected_ut_set = set()

        def window_to_str(row):
            ''' strings are used (stored as a set) to make the lookup fast and easy '''
            return '-'.join(str(v) for v in row.values)

        def was_detected(row):
            window_str = window_to_str(row)
            # logging.info(f'window_str={window_str}')
            return window_str  in detected_ut_set

        for i, row in detected_ut.iterrows():
            detected_ut_set.add( window_to_str(row) )

        # get PC values in abnormal run where unseen transitions (detected_ut) are observed
        # col_a_detected_points = col_a[ col_a.iloc[:,0].rolling(self.train_n).apply(lambda x: was_detected(x) ) > 0.0 ]

        # col_a['detected_anomaly'] = col_a.iloc[:,0].rolling(self.train_n).apply(lambda x: was_detected(x) ) > 0.0
        # return col_a
        results = col_a.iloc[:,0].rolling(self.train_n).apply(lambda x: was_detected(x) ).dropna() > 0.0
        return results

    def predict_all(self, df_a):
        return [self.predict(df_a[col_a]) for col_a in df_a]

    #def evaluate(self, col_a, col_a_ground_truth):
    #    ''' results = return of predict function '''
    #    # cm = confusion_matrix(col_a['detected_anomaly'].values, col_a_ground_truth.values)

    #    # support means how many normal/anomalous datapoints there were
    #    precision, recall, fscore, support = precision_recall_fscore_support(col_a['detected_anomaly'].values, col_a_ground_truth.values)
    #    return precision, recall, fscore, support 


    def evaluate_all(self, results_all, df_a_ground_truth_windowized):
        ''' results_all = return of predict_all function 
            This function returns 2 evaluation metrics that really matter
            for anomaly detection systems:
            - anomaly recall
            - false anomalies (referred to as "false positives" in some papers)
        '''
        # results = []
        # for col_a, col_a_name in zip(results_all, df_a_ground_truth):
        #     result = self.evaluate(col_a, df_a_ground_truth[col_a_name])
        #     results.append(result)

        
        # windowize the ground truth labels (so they determine if the whole window/sequence was anomalous)
        # 

        # concatinate detection results and ground truth labels from 
        # all test examples (in other words, flatten nested list)
        all_detection_results = [val for results in results_all for val in results]
        all_ground_truth = df_a_ground_truth_windowized.melt(value_name='melted').drop('variable', axis=1).dropna()[['melted']].values.reshape(-1).tolist()
       
        # all_detection_results[0] = True # TODO: DELETE (it allowed verifying correctness of evaluation metrics)

        precision, recall, fscore, support = precision_recall_fscore_support(all_ground_truth, all_detection_results)

        # what percent of anomalies will get detected
        anomaly_recall = recall[1]
        # what percent of normal program behaviour will be classified as anomalous
        inverse_normal_recall = 1 - recall[0]
        return anomaly_recall, inverse_normal_recall

#         gt_counts = all_ground_truth.value_counts()
#         pred_counts = all_detection_results.value_counts()
# 
#         # 4 numbers that can calculate any evaluation metrics
#         gt_normal_count = gt_counts[False]
#         gt_anomalous_count = gt_counts[True]
#         pred_normal_count = pred_counts[False]
#         pred_anomalous_count = pred_counts[True]
# 
#         # all_detection_results = all_detection_results.values.tolist()
#         # all_ground_truth = all_ground_truth.values.tolist()
# 
#         return anomal, normal_classified_as_anomaly
# 

#def detect(df_n, df_a, n=2):
#    utils.print_header(f'UNIQUE TRANSITIONS (n={n})')
#    normal_ut = utils.pc_df_to_sliding_windows(df_n, window_size=n, unique=True)
#    abnormal_ut = utils.pc_df_to_sliding_windows(df_a, window_size=n, unique=True)
#
#    logging.info(f'Number of train programs: {df_n.shape[1]}')
#    logging.info(f'Longest train program size: {df_n.shape[0]} instructions')
#    logging.info(f'Number of unique train sequences (with size of {n}): {normal_ut.shape[0]}')
#
#    # gets abnormal_ut entries that are not present in normal_ut
#    # (it ignores df index so that's why it's ugly)
#    # it's from: https://stackoverflow.com/a/50645672/4620679
#    # import pdb; pdb.set_trace()
#    detected_ut = abnormal_ut[ ~abnormal_ut[ ~abnormal_ut.stack().isin(normal_ut.stack().values).unstack()].isna().all(axis=1) ].dropna()
#   
#    # set is used for fast lookup
#    detected_ut_set = set()
#    for i, row in detected_ut.iterrows():
#        detected_ut_set.add('-'.join(str(v) for v in row.values))
#
#    def was_detected(row):
#        return '-'.join(str(v) for v in row.values) in detected_ut_set
#
#    # At this point, if detected_ut is not empty, it means that abnormal behaviour was detected.
#    # We can further plot where exactly it happened during program execution.
#
#    # get PC values in abnormal run where unseen transitions (detected_ut) are observed
#    # df_a_col0 =  df_a[df_a.columns[0]]
#    # df_a_detected_points = df_a[ df_a.iloc[:,0].rolling(2).apply(lambda x: ((detected_ut['all_pc'] == x.iloc[0]) & (detected_ut['all_pc_shifted'] == x.iloc[1])).any() ) > 0.0 ]
#    df_a_detected_points = df_a[ df_a.iloc[:,0].rolling(n).apply(lambda x: was_detected(x) ) > 0.0 ]
#
#    logging.info(f'Test program size: {df_a.shape[0]} instructions')
#    logging.info(f'Number of detected anomalies in test program: {df_a_detected_points.shape[0]}')
#    return detected_ut, df_a_detected_points


if __name__ == '__main__':
    # function_ranges are used just for plotting
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

    n = 3

    function_ranges = json.load(args.function_ranges) if args.function_ranges else {}
    df_n = df_from_pc_files(args.normal_pc, column_prefix='normal: ')
    df_a = df_from_pc_files(args.abnormal_pc, column_prefix='abnormal: ')
    # windows4 = utils.pc_df_to_sliding_windows(df_n, window_size=4, unique=True)
    # windows10 = utils.pc_df_to_sliding_windows(df_n, window_size=10, unique=True)

    ax  = plot_pc_timeline(df_n, function_ranges)
    ax2 = plot_pc_timeline(df_a, function_ranges)

    detected_ut, df_a_detected_points = detect(df_n, df_a, n=n)
    df_a_detected_points.plot(ax=ax2, color='r', marker='*', markersize=10, linestyle='none', legend=None)

    utils.plot_vspans(ax2, df_a_detected_points.index.values - n, n)

    plt.show()
    import pdb; pdb.set_trace()
    

