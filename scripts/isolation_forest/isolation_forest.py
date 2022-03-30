#!/usr/bin/python3.7

###### Step 1: Import Libraries
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import IsolationForest

import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import json
import os
import inspect
from sklearn.metrics import precision_recall_fscore_support

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import logging

import utils
from utils import read_pc_values, plot_pc_histogram, plot_pc_timeline, df_from_pc_files
from detection_model import Detection_Model

# if_model = IsolationForest(n_estimators=100, random_state=0).fit(X_train)
# # Predict the anomalies
# if_prediction = if_model.predict(X_test)
# # Change the anomalies' values to make it consistent with the true values
# if_prediction = [1 if i==-1 else 0 for i in if_prediction]
# # Check the model performance
# print(classification_report(y_test, if_prediction))
# # Train the isolation forest model
# if_model = IsolationForest(n_estimators=100, random_state=0, warm_start=True).fit(X_train)
# # Use new data to train 50 trees on top of existing model 
# if_model.n_estimators += 50
# if_model.fit(X_more)
# # Predict the anomalies
# if_prediction = if_model.predict(X_test)
# # Change the anomalies' values to make it consistent with the true values
# if_prediction = [1 if i==-1 else 0 for i in if_prediction]
# # Check the model performance
# print(classification_report(y_test, if_prediction))
# ###### Step 6: Isolation Forest With Warm Start On New Trees
# # Train the isolation forest model
# if_model = IsolationForest(n_estimators=100, random_state=0, warm_start=True).fit(X_train)
# # Use the existing data to train 20 trees on top of existing model 
# if_model.n_estimators += 20
# if_model.fit(X_train)
# # Predict the anomalies
# if_prediction = if_model.predict(X_test)
# # Change the anomalies' values to make it consistent with the true values
# if_prediction = [1 if i==-1 else 0 for i in if_prediction]
# # Check the model performance
# print(classification_report(y_test, if_prediction))


class Isolation_Forest(Detection_Model):
    def __init__(self, *args, **kwargs):
        self.model = IsolationForest(n_estimators=100, random_state=0, warm_start=True, *args, **kwargs)
        self.train_n = None

    def train(self, df_n, n=2):
        utils.print_header(f'ISOLATION FOREST (n={n})')
        normal_windows = utils.pc_df_to_sliding_windows(df_n, window_size=n, unique=True)
        self.model.fit(normal_windows)
        self.train_n = n

        logging.info(f'Number of train programs: {df_n.shape[1]}')
        logging.info(f'Longest train program size: {df_n.shape[0]} instructions')
        logging.info(f'Number of unique train sequences (with size of {n}): {normal_windows.shape[0]}')

    def predict(self, df_a_col):
        # gets abnormal_ut entries that are not present in normal_ut
        # (it ignores df index so that's why it's ugly)
        # it's from: https://stackoverflow.com/a/50645672/4620679
        # import pdb; pdb.set_trace()

        # logging.info(df_a_col)
        # logging.info(type(df_a_col))

        abnormal_windows = utils.pc_df_to_sliding_windows(df_a_col, window_size=self.train_n, unique=False)
        results = self.model.predict(abnormal_windows)
        results = [i==-1 for i in results]
        return results

    # def predict_all(self, df_a):
    #     return [self.predict(df_a[[col_a]]) for col_a in df_a]


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
    

