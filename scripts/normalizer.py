import tensorflow as tf
import pandas as pd
import numpy as np

class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def assign_min_max_for_normalization(self, X_train):
        self.min_val = tf.reduce_min(X_train.values)
        self.max_val = tf.reduce_max(X_train.values)

    def normalize(self, X):
        if type(X) == pd.core.frame.DataFrame:
            X = X.values
        X = (X - self.min_val) / (self.max_val - self.min_val)
        return X

