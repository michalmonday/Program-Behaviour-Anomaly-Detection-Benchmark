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

# 3 lines below allow using imports from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import logging

import utils
from utils import read_pc_values, plot_pc_histogram, plot_pc_timeline, df_from_pc_files
from detection_model import Detection_Model

# all detection methods must inherit from the Detection_Model class (which provides evaluation consistency)
class Isolation_Forest(Detection_Model):
    def __init__(self, *args, **kwargs):
        ''' We can initialize our detection method however we want, in this case 
        it creates "model" member holding reference to IsolationForest from scikit-learn.
        It isn't required to have "model" member name or any members at all being initialized. '''
        self.model = IsolationForest(n_estimators=100, random_state=0, warm_start=True, *args, **kwargs)

    def train(self, normal_windows, **kwargs):
        # normal_windows is a 2D numpy array where each row contains input features of a single example
        self.model.fit(normal_windows)

    def predict(self, abnormal_windows):
        ''' This method must return a list of boolean values, 
        one for each row of "abnormal_windows" 2D numpy array. '''
        # abnormal_windows is a 2D numpy array where each row contains input features of a single example
        return [i==-1 for i in self.model.predict(abnormal_windows)]

