#!/usr/bin/python3.7

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.svm import OneClassSVM
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


class OneClass_SVM(Detection_Model):
    def __init__(self, nu=0.01, kernel='rbf', gamma='auto', *args, **kwargs):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, *args, **kwargs)
        self.train_n = None

    def train(self, df_n, n=2):
        # utils.print_header(f'ONE CLASS SVM (n={n})')
        normal_windows = utils.pc_df_to_sliding_windows(df_n, window_size=n, unique=True)
        self.model.fit(normal_windows)
        self.train_n = n

        # logging.info(f'Number of train programs: {df_n.shape[1]}')
        # logging.info(f'Longest train program size: {df_n.shape[0]} instructions')
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


