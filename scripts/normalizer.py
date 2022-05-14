import tensorflow as tf
import pandas as pd
import numpy as np

class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def assign_min_max_for_normalization(self, X_train):
        self.min_val = tf.reduce_min(X_train.values).numpy()
        self.max_val = tf.reduce_max(X_train.values).numpy()

    def normalize(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val)

