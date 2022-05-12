#!/usr/bin/python3.7

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.neighbors import LocalOutlierFactor
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


class Local_Outlier_Factor(Detection_Model):
    def __init__(self, *args, **kwargs):
        self.model = LocalOutlierFactor(n_neighbors=5, novelty=True, *args, **kwargs)
        self.train_n = None

    def train(self, normal_windows):
        self.model.fit(normal_windows)

    def predict(self, abnormal_windows):
        return [i==-1 for i in self.model.predict(abnormal_windows)]


