#!/usr/bin/python3.7
# coding: utf-8

'''
What's the difference between unique transition methods, lstm autoencoder and "conventional" 
machine learning methods?

Unique transitions and lstm autoencoder do not use anomalous examples for training.
Meaning that they will attempt to detect any type of anomaly, even previously unseen.

What I call here "conventional" machine learning methods, use both, normal and abnormal
examples for training. 
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
pd.options.mode.chained_assignment = None
# import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import glob
# from livelossplot import PlotLossesKeras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import utils
from copy import deepcopy
import logging
from utils import read_pc_values, df_from_pc_files, plot_pc_timeline
from detection_model import Detection_Model

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

models = [
    # MultinomialNB,
    # BernoulliNB,
    LogisticRegression,
    SGDClassifier,
    SVC,
    LinearSVC
    ]

min_val = None
max_val = None


# TODO: For training, supply a set of labeled examples including normal and abnormal.

def assign_min_max_for_normalization(X_train):
    global min_val
    global max_val
    min_val = tf.reduce_min(X_train.values)
    max_val = tf.reduce_max(X_train.values)

def normalize(X):
    global min_val
    global max_val
    if type(X) == pd.core.frame.DataFrame:
        X = X.values
    X = (X - min_val) / (max_val - min_val)
    # X = tf.cast(X, tf.float32)
    return X

#class Conventional_Model(Detection_Model):
#    ''' Detection_Model contains evaluation methods '''
#    def __init__(self, sklearn_model):
#        self.sklearn_model = sklearn_model
#        # list of min/max values for each autoencoder (within a forest)
#        self.min_val = None
#        self.max_val = None
#        self.window_size = None

#    def assign_min_max_for_normalization(self, X_train):
#        ''' There are multiple lists instead of single values because 
#            "autoencoder forest" method may be used, where multiple 
#            models are created (like an array of models), and each model
#            has its own normalization. '''
#        self.min_val = tf.reduce_min(X_train.values)
#        self.max_val = tf.reduce_max(X_train.values)

#    def train(self, df_n, window_size=20, epochs=10):
#        # keep reference to supplied window size, to use the same at testing/predicting
#        self.window_size = window_size
#        utils.print_header(f'{self.sklearn_model.__class__.__name__} (window_size={self.window_size}, number_of_models={number_of_models})')
#        X_train = utils.pc_df_to_sliding_windows(df_n, self.window_size, unique=True)
#        self.assign_min_max_for_normalization(X_train)
#        self.std_ranges = self.get_std_ranges(X_train, number_of_models)

#            X_train = self.normalize(X_train)
#            X_train = np.array( X_train ).reshape(-1, self.window_size, 1)
#            self.sklearn_model.fit(X_train)

#                self.models.append(model)
#                history = model.fit(
#                    X_train, X_train,#y_train, #X_train
#                    epochs=epochs,
#                    batch_size=5000,#32,
#                    # validation_split=0.1,
#                    shuffle=True,
#                    verbose=0 # 0=silent, 1=progress bar, 2=one line per epoch
#                    # callbacks=[PlotLossesKeras()]
#                )
#                X_train_pred = model(X_train)
#                train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
#                self.thresholds.append( train_mae_loss.max() )
#                # X_test_pred = model(X_test)
#                # test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

#            else:
#                # None as a model means that any testing examples 
#                # will be classified as anomalies
#                self.models.append(None)

#                # if no training examples were in current std_range
#                # but some testing examples were, then all testing examples
#                # are anomalous
#                logging.info(f'None of training windows had standard deviation in {std_range} range. But some testing windows had, all of them are going to be classified as anomalous (test_mae_loss is forced to be 0.001 and threshold is forced to be 0). subset_indices={subset_indices}')
#                self.thresholds.append(0.0)
#        # model.save('lstm_autoencoder_model.h5')
#        # logging.info('\nExample model.summary():')
#        # model.summary()

#    def predict(self, df_a):
#        df_a = df_a.dropna()
#        X_test = utils.pc_df_to_sliding_windows(df_a, self.window_size)
#        results_df = pd.DataFrame(np.NaN, index=df_a.index.values[:-self.window_size+1] , columns = ['loss', 'threshold', 'anomaly', 'window_start', 'window_end'])
#        for i, std_range in enumerate(self.std_ranges):
#            model = self.models[i]
#            threshold = self.thresholds[i]
#            X_test = get_windows_subset(X_test, std_range)
#            if X_test_subset.empty:
#                continue
#            subset_indices = X_test_subset.index.values
#            X_test = np.array( X_test ).reshape(-1, self.window_size, 1)
#            # normalize
#            X_test = self.normalize(X_test)
#            if model:
#                X_test_pred = model(X_test)
#                test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
#            else:
#                test_mae_loss = np.array([0.001]*X_test.shape[0])
#            results_df['loss'].loc[subset_indices] = test_mae_loss.reshape(-1)
#            results_df['threshold'].loc[subset_indices] = threshold
#            results_df['anomaly'].loc[subset_indices] = results_df['loss'].loc[subset_indices] > results_df['threshold'].loc[subset_indices]
#            results_df['window_start'].loc[subset_indices] = results_df.loc[subset_indices].index.values
#            results_df['window_end'].loc[subset_indices] = results_df['window_start'].loc[subset_indices] + self.window_size
#        # anomalies = results_df[results_df.anomaly == True]
#        # is_anomalous = not anomalies.empty
#        # return is_anomalous, results_df, anomalies
#        results_df_not_filled = results_df[ results_df.isna().any(axis=1) ]
#        if not results_df_not_filled.empty:
#            logging.error(f'Some results rows were not filled...')
#            logging.error(results_df_not_filled)
#            import pdb; pdb.set_trace()

#        return results_df.anomaly.dropna().values.tolist()

#    def predict_all(self, df_a):
#        return [self.predict(df_a[col_a]) for col_a in df_a]

#    #def evaluate_all(self, results_all, df_a_ground_truth_windowized):
#    #    ''' results_all = return of predict_all function 
#    #        This function returns 2 evaluation metrics that really matter
#    #        for anomaly detection systems:
#    #        - anomaly recall
#    #        - false anomalies (referred to as "false positives" in some papers)
#    #    '''
#    #    # results = []
#    #    # for col_a, col_a_name in zip(results_all, df_a_ground_truth):
#    #    #     result = self.evaluate(col_a, df_a_ground_truth[col_a_name])
#    #    #     results.append(result)

#    #    
#    #    # windowize the ground truth labels (so they determine if the whole window/sequence was anomalous)
#    #    # 

#    #    # concatinate detection results and ground truth labels from 
#    #    # all test examples (in other words, flatten nested list)
#    #    all_detection_results = [val for results in results_all for val in results]
#    #    all_ground_truth = df_a_ground_truth_windowized.melt(value_name='melted').drop('variable', axis=1).dropna()[['melted']].values.reshape(-1).tolist()
#    #   
#    #    # all_detection_results[0] = True # TODO: DELETE (it allowed verifying correctness of evaluation metrics)

#    #    # import pdb; pdb.set_trace()
#    #    precision, recall, fscore, support = precision_recall_fscore_support(all_ground_truth, all_detection_results)

#    #    # what percent of anomalies will get detected
#    #    anomaly_recall = recall[1]
#    #    # what percent of normal program behaviour will be classified as anomalous
#    #    inverse_normal_recall = 1 - recall[0]
#    #    return anomaly_recall, inverse_normal_recall

##def detect(df_n, df_a, window_size=20, epochs=10, number_of_models=6):
##    utils.print_header(f'LSTM AUTOENCODER (window_size={window_size}, number_of_models={number_of_models})')
##
##    # for training data duplicate windows are dropped
##    # it greatly improves training times
##
##    # scaler = StandardScaler()
##    # melted_df_n = df_n.melt(value_name='all_pc').drop('variable', axis=1).dropna()[['all_pc']].values.reshape(-1,1)
##    # scaler.fit(melted_df_n)
##    # import pdb; pdb.set_trace()
##    # df_n_scaled = df_n.apply(scaler.transform, axis=1)
##    # df_a_scaled = df_a.apply(scaler.transform, axis=1)
##
##    X_train = utils.pc_df_to_sliding_windows(df_n, window_size, unique=True)
##    X_test = utils.pc_df_to_sliding_windows(df_a, window_size)
##
##    results_df = pd.DataFrame( index=df_a.index.values[:-window_size+1] , columns = ['loss', 'threshold', 'anomaly', 'window_start', 'window_end'])
##    std_train = X_train.std(axis=1)
##    std_test = X_test.std(axis=1)
##
##    ranges = get_std_ranges(X_train, number_of_models)
##    logging.info(f'The following {number_of_models} standard deviation ranges are going to be used:')
##    for std_range in ranges:
##        logging.info(std_range)
##
##    logging.info(f'\nTraining and testing {number_of_models} LSTM autoencoders:')
##    logging.info(f'   {"std range":<15} {"train":<5} {"test":<5}')
##    for i, std_range in enumerate(ranges):
##        # for testing data speed isn't a problem (predictions are done relatively fast)
##        # so duplicates don't have to be dropped (which is good because it wouldn't be good for presenting results)
##        X_train_subset = get_windows_subset(X_train, std_range)
##        X_test_subset = get_windows_subset(X_test, std_range)
##        subset_indices = X_test_subset.index.values
##
##        X_train_subset, X_test_subset = normalize(X_train_subset, X_test_subset)
##
##        # y_train = produce_y( X_train_subset )
##        # X_train_subset = np.array( X_train_subset[:-1] ).reshape(-1, window_size, 1)
##        X_train_subset = np.array( X_train_subset ).reshape(-1, window_size, 1)
##        # y_test = produce_y( X_test_subset )
##        # X_test_subset = np.array( X_test_subset[:-1] ).reshape(-1, window_size, 1)
##        X_test_subset = np.array( X_test_subset ).reshape(-1, window_size, 1)
##
##        print_table_row(i, std_range, X_train_subset.shape[0], X_test_subset.shape[0])
##
##        model = create_model(X_train_subset)
##
##        # import pdb; pdb.set_trace()
##
##        # if there isn't any testing windows having this std range
##        # then there's no need to train the model
##        if not X_test_subset.any():
##            logging.info(f'None of testing windows had standard deviation in range {std_range}')
##            continue
##
##        # if there are testing examples, but not training ones,
##        # then model can't be trained and all testing examples
##        # should be marked as anomalous (because they were probably
##        # not present in training)
##        if X_train_subset.any():
##            history = model.fit(
##                X_train_subset, X_train_subset,#y_train, #X_train_subset
##                epochs=epochs,
##                batch_size=5000,#32,
##                # validation_split=0.1,
##                shuffle=True,
##                verbose=0 # 0=silent, 1=progress bar, 2=one line per epoch
##                # callbacks=[PlotLossesKeras()]
##            )
##
##            #plt.show()
##            #plt.plot(history.history['loss'], label='train')
##            #plt.plot(history.history['val_loss'], label='test')
##            #plt.legend();
##            #plt.show()
##
##            # X_train_pred = model.predict(X_train_subset)
##            # import pdb;pdb.set_trace()
##            X_train_pred = model(X_train_subset)
##            train_mae_loss = np.mean(np.abs(X_train_pred - X_train_subset), axis=1)
##            # X_test_pred = model.predict(X_test_subset)
##            X_test_pred = model(X_test_subset)
##            test_mae_loss = np.mean(np.abs(X_test_pred - X_test_subset), axis=1)
##            THRESHOLD = train_mae_loss.max()
##
##        else:
##            # if no training examples were in current std_range
##            # but some testing examples were, then all testing examples
##            # are anomalous
##            logging.info(f'None of training windows had standard deviation in {std_range} range. But some testing windows had, all of them are going to be classified as anomalous (test_mae_loss is forced to be 0.001 and threshold is forced to be 0). subset_indices={subset_indices}')
##            test_mae_loss = np.array([0.001]*X_test_subset.shape[0])
##            THRESHOLD = 0.0
##
##        # results_df = pd.DataFrame(index=test[window_size:].index) #(index=test[window_size:].index)
##
##        # results_df = pd.DataFrame( index=df_a.index.values[window_size:-window_size] )
##        
##        # results_df['loss'] = np.concatenate((train_mae_loss, test_mae_loss))
##        results_df['loss'].loc[subset_indices] = test_mae_loss.reshape(-1)
##        results_df['threshold'].loc[subset_indices] = THRESHOLD
##        results_df['anomaly'].loc[subset_indices] = results_df['loss'].loc[subset_indices] > results_df['threshold'].loc[subset_indices]
##        results_df['window_start'].loc[subset_indices] = results_df.loc[subset_indices].index.values
##        results_df['window_end'].loc[subset_indices] = results_df['window_start'].loc[subset_indices] + window_size
##
##    # results_df['close'] = test[window_size:].close
##    # results_df[['train_loss', 'test_loss', 'threshold', 'anomaly']].plot()
##
##
##    #plt.plot(results_df.index, results_df.loss, label='loss')
##    #plt.plot(results_df.index, results_df.threshold, label='threshold')
##    #plt.xticks(rotation=25)
##    #plt.legend()
##    #plt.show()
##
##    anomalies = results_df[results_df.anomaly == True]
##    logging.info(f'Number of detected anomalies in test program: {anomalies.shape[0]}')
##
##    model.save('lstm_autoencoder_model.h5')
##    logging.info('\nExample model.summary():')
##    model.summary()
##
##    return results_df, anomalies


#def plot_results(df_a, results_df, anomalies_df, window_size, fig_title='', function_ranges={}):
#    fig, axs = plt.subplots(2)
#    # fig.subplots_adjust(top=0.92, hspace=0.43)
#    fig.subplots_adjust(hspace=0.43, top=0.835)
#    if fig_title:
#        fig.suptitle(fig_title, fontsize=20)
#    ax = results_df[['loss']].plot(ax=axs[0], color='purple', linewidth=0.7)
#    # markers to show distinct points
#    results_df[['loss']].plot(ax=axs[0], marker='h', markersize=1, linestyle='none', legend=None)
#    results_df[['threshold']].plot(ax=axs[0], color='red', legend=None)

#    ax.set_title(f'Reconstruction error of each window (size={window_size}) in compromised program')
#    ax.set_xlabel('Index of sliding window')
#    ax.set_ylabel('Window reconstruction error')
#    ax.legend().remove()

#    # threshold = results_df['threshold'].iloc[0]
#    last_threshold = results_df['threshold'].iloc[-1]
#    # ax.axhline(threshold, color='r', label='Threshold')#, linestyle='--')

#    ax_t = ax.twinx()
#    ax_t.set_yticks([last_threshold])
#    ax_t.set_yticklabels(['Threshold'], fontdict={'fontsize':7})
#    ax_t.set_ylim(*ax.get_ylim())
#    ax_t.legend().remove()
#    # draw line to show where abnormal values actually were
#    # ax.axvline(results_df.index.values[4905-window_size], color='k', linestyle='--', label='normal vs abnormal')
#    plot_pc_timeline(df_a, ax=axs[1], function_ranges=function_ranges)
#    # draw line to show where abnormal values actually were
#    # axs[1].axvline(results_df.index.values[4905-window_size], color='k', linestyle='--')
#    axs[1].set_title(f'Results ({anomalies_df.shape[0]} anomalous windows were detected)')
#    axs[1].set_xlabel('Instruction index')
#    axs[1].set_ylabel('Program counter (address)')
#    axs[1].get_yaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')

#    if not anomalies_df.shape[0]:
#        return axs
#    # draw anomaly region highlights
#    for i, row in anomalies_df.iterrows():
#        axs[1].axvspan(row['window_start'], row['window_end'], color='red', alpha=0.15)
#    # draw stars
#    df_a.iloc[ anomalies_df['window_end'] ].plot(ax=axs[1], color='r', marker='*', markersize=10, linestyle='none', legend=None)
#    return axs



#if __name__ == '__main__':
#    logging.getLogger().setLevel(logging.INFO)
#    window_size = 20

#    init_settings_i_dont_know()

#    normal_f_names = list(glob.glob('../../log_files/*normal*pc'))
#    anomaly_f_names = list(glob.glob('../../log_files/*compromised*pc'))

#    # use first files only just for simplicity 
#    # df_n = df_from_pc_files(normal_f_names).iloc[:,0]
#    # df_a = df_from_pc_files(anomaly_f_names).iloc[:,0]

#    # each column comes from a different file
#    # column name = file name
#    # row = program counters from different files
#    df_n = df_from_pc_files(normal_f_names) 
#    df_a = df_from_pc_files(anomaly_f_names)

#    results_df, anomalies_df = detect(df_n, df_a, window_size=window_size)
#    plot_results(df_a, results_df, anomalies_df, window_size)
#    # plt.legend();
#    plt.show()

#    # normal_f_names = list(glob.glob('../../log_files/*mimic*pc'))
#    # anomaly_f_names = list(glob.glob('../../log_files/*normal.pc'))


