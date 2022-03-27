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
from detection_model import Detection_Model

class Freq_Dist(Detection_Model):
    def __init__(self):
        pass
    
    def train(self, df_n):
        pass

    def predict(self, col_a):
        pass

    def predict_all(self, df_a):
        return [self.predict(df_a[[col_a]]) for col_a in df_a]


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
    


