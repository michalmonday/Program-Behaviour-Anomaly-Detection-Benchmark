import tensorflow as tf
from tensorflow.keras import datasets, layers, models
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


def create_model(window_size, size_multiplier=1):
    model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(window_size, 1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(2))
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    model.add(layers.Conv1D(64 * size_multiplier, 2, activation="relu", input_shape=(window_size,1)))
    model.add(layers.Dense(16 * size_multiplier, activation="relu"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', 
            optimizer = "adam",               
            metrics = tf.keras.metrics.SparseCategoricalAccuracy(
                  name="sparse_categorical_accuracy", dtype=None
              )
        )
    return model


class CNN(Detection_Model):
    def __init__(self, *args, **kwargs):
        self.network_size_multiplier = kwargs.get('network_size_multiplier', 1)

    def train(self, windows, labels=None, epochs=1, **kwargs):
        if labels is None:
            raise Exception('No labels were supplied to CNN.train(...) method.')
        self.model = create_model(windows.shape[1], size_multiplier=self.network_size_multiplier)
        self.model.fit(windows, labels, epochs=epochs)

    def predict(self, X_test):
        return self.model.predict(X_test)[:,0] < 0.5


