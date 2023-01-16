# import tensorflow as tf
import pandas as pd
import numpy as np

class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def assign_min_max_for_normalization(self, X_train):
        # import pdb; pdb.set_trace()
        self.min_val = X_train.values.min() 
        self.max_val = X_train.values.max() 

    def normalize(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val)

