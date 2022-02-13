#!/usr/bin/python3.7
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
# import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import glob
# from livelossplot import PlotLossesKeras
from sklearn.preprocessing import StandardScaler

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import utils
from copy import deepcopy
from utils import read_pc_values, df_from_pc_files, plot_pc_timeline
import logging


def init_settings_i_dont_know():
    register_matplotlib_converters()
    # sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    # rcParams['figure.figsize'] = 22, 10
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

def produce_y(X_windows):
    ''' y is the first program counter value following the last 
        program counter of a window. That's how it was produced in the youtube
        tutorial:
        https://www.youtube.com/watch?v=H4J74KstHTE

        However, in another example, "y" was simply X, which is natural
        for autoencoders:
        https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb

        From my observations, training with "y" equal to X produces the same
        results but takes more time.
        '''
    ys = []
    for i, window in enumerate(X_windows):
        if i == 0:
            continue
        ys.append(window[-1]) 
    return np.array(ys)

# def temporalize(X, y, lookback):
#     output_X = []
#     output_y = []
#     for i in range(len(X)-lookback-1):
#         t = []
#         for j in range(1,lookback+1):
#             # Gather past records upto the lookback period
#             t.append(X[[(i+j+1)], :])
#         output_X.append(t)
#         output_y.append(y[i+lookback+1])
#     return output_X, output_y

from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def create_model(X_train):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=64, 
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')
    return model

def normalize(X_train, X_test):
    X_train = X_train.values
    X_test = X_test.values
    min_val = tf.reduce_min(X_train)
    max_val = tf.reduce_max(X_train)
    X_train = (X_train - min_val) / (max_val - min_val)
    X_test = (X_test - min_val) / (max_val - min_val)

    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)
    return X_train, X_test

def get_std_ranges(X_train, number_of_models, epsilon=0.00001):
    ''' ranges to create a forest of autoencoders '''
    std = X_train.std(axis=1)
    #std_interval = (std.max() - std.min()) / number_of_models
    #ranges = list(zip(
    #        np.arange(std.min(), std.max(), std_interval),
    #        np.arange(std.min() + std_interval, std.max() + epsilon, std_interval)
    #        ))
    ranges = []
    for interval in pd.qcut( pd.DataFrame(std)[0], number_of_models, duplicates='drop').unique():
        ranges.append((
            interval.left, 
            interval.right
            ))
    ranges = sorted(ranges)
    ranges[0] = (0.0, ranges[0][1])
    ranges[-1] = (ranges[-1][0], 99999999.0)
    return ranges

def get_windows_subset(windows, std_range):
    std = windows.std(axis=1)
    return windows[(std >= std_range[0]) & (std <= std_range[1])]

def print_table_row(range_index, std_range, train_windows_count, test_windows_count):
    std_range_str = f'{round(std_range[0],1)} - {round(std_range[1],1)}'
    line = f'{range_index+1:<3}{std_range_str:<15} {train_windows_count:<5} {test_windows_count:<5}'

    logging.info(line)

def detect(df_n, df_a, window_size=20, epochs=10, number_of_models=6):
    utils.print_header(f'LSTM AUTOENCODER (window_size={window_size}, number_of_models={number_of_models})')

    # for training data duplicate windows are dropped
    # it greatly improves training times

    # scaler = StandardScaler()
    # melted_df_n = df_n.melt(value_name='all_pc').drop('variable', axis=1).dropna()[['all_pc']].values.reshape(-1,1)
    # scaler.fit(melted_df_n)
    # import pdb; pdb.set_trace()
    # df_n_scaled = df_n.apply(scaler.transform, axis=1)
    # df_a_scaled = df_a.apply(scaler.transform, axis=1)

    X_train = utils.pc_df_to_sliding_windows(df_n, window_size, unique=True)
    X_test = utils.pc_df_to_sliding_windows(df_a, window_size)

    results_df = pd.DataFrame( index=df_a.index.values[:-window_size+1] , columns = ['loss', 'threshold', 'anomaly', 'window_start', 'window_end'])
    std_train = X_train.std(axis=1)
    std_test = X_test.std(axis=1)

    ranges = get_std_ranges(X_train, number_of_models)
    logging.info(f'The following {number_of_models} standard deviation ranges are going to be used:')
    for std_range in ranges:
        logging.info(std_range)

    logging.info(f'\nTraining and testing {number_of_models} LSTM autoencoders:')
    logging.info(f'   {"std range":<15} {"train":<5} {"test":<5}')
    for i, std_range in enumerate(ranges):
        # for testing data speed isn't a problem (predictions are done relatively fast)
        # so duplicates don't have to be dropped (which is good because it wouldn't be good for presenting results)
        X_train_subset = get_windows_subset(X_train, std_range)
        X_test_subset = get_windows_subset(X_test, std_range)
        subset_indices = X_test_subset.index.values

        X_train_subset, X_test_subset = normalize(X_train_subset, X_test_subset)

        # y_train = produce_y( X_train_subset )
        # X_train_subset = np.array( X_train_subset[:-1] ).reshape(-1, window_size, 1)
        X_train_subset = np.array( X_train_subset ).reshape(-1, window_size, 1)
        # y_test = produce_y( X_test_subset )
        # X_test_subset = np.array( X_test_subset[:-1] ).reshape(-1, window_size, 1)
        X_test_subset = np.array( X_test_subset ).reshape(-1, window_size, 1)

        print_table_row(i, std_range, X_train_subset.shape[0], X_test_subset.shape[0])

        model = create_model(X_train_subset)

        # import pdb; pdb.set_trace()

        # if there isn't any testing windows having this std range
        # then there's no need to train the model
        if not X_test_subset.any():
            logging.info(f'None of testing windows had standard deviation in range {std_range}')
            continue

        # if there are testing examples, but not training ones,
        # then model can't be trained and all testing examples
        # should be marked as anomalous (because they were probably
        # not present in training)
        if X_train_subset.any():
            history = model.fit(
                X_train_subset, X_train_subset,#y_train, #X_train_subset
                epochs=epochs,
                batch_size=5000,#32,
                # validation_split=0.1,
                shuffle=True,
                verbose=0 # 0=silent, 1=progress bar, 2=one line per epoch
                # callbacks=[PlotLossesKeras()]
            )

            #plt.show()
            #plt.plot(history.history['loss'], label='train')
            #plt.plot(history.history['val_loss'], label='test')
            #plt.legend();
            #plt.show()

            # X_train_pred = model.predict(X_train_subset)
            X_train_pred = model(X_train_subset)
            train_mae_loss = np.mean(np.abs(X_train_pred - X_train_subset), axis=1)
            # X_test_pred = model.predict(X_test_subset)
            X_test_pred = model(X_test_subset)
            test_mae_loss = np.mean(np.abs(X_test_pred - X_test_subset), axis=1)
            THRESHOLD = train_mae_loss.max()

        else:
            # if no training examples were in current std_range
            # but some testing examples were, then all testing examples
            # are anomalous
            logging.info(f'None of training windows had standard deviation in {std_range} range. But some testing windows had, all of them are going to be classified as anomalous (test_mae_loss is forced to be 0.001 and threshold is forced to be 0). subset_indices={subset_indices}')
            test_mae_loss = np.array([0.001]*X_test_subset.shape[0])
            THRESHOLD = 0.0

        # results_df = pd.DataFrame(index=test[window_size:].index) #(index=test[window_size:].index)

        # results_df = pd.DataFrame( index=df_a.index.values[window_size:-window_size] )
        
        # results_df['loss'] = np.concatenate((train_mae_loss, test_mae_loss))
        results_df['loss'].loc[subset_indices] = test_mae_loss.reshape(-1)
        results_df['threshold'].loc[subset_indices] = THRESHOLD
        results_df['anomaly'].loc[subset_indices] = results_df['loss'].loc[subset_indices] > results_df['threshold'].loc[subset_indices]
        results_df['window_start'].loc[subset_indices] = results_df.loc[subset_indices].index.values
        results_df['window_end'].loc[subset_indices] = results_df['window_start'].loc[subset_indices] + window_size

    # results_df['close'] = test[window_size:].close
    # results_df[['train_loss', 'test_loss', 'threshold', 'anomaly']].plot()


    #plt.plot(results_df.index, results_df.loss, label='loss')
    #plt.plot(results_df.index, results_df.threshold, label='threshold')
    #plt.xticks(rotation=25)
    #plt.legend()
    #plt.show()

    anomalies = results_df[results_df.anomaly == True]
    logging.info(f'Number of detected anomalies in test program: {anomalies.shape[0]}')

    model.save('lstm_autoencoder_model.h5')
    logging.info('\nExample model.summary():')
    model.summary()

    return results_df, anomalies


def plot_results(df_a, results_df, anomalies_df, window_size, fig_title='', function_ranges={}):
    fig, axs = plt.subplots(2)
    # fig.subplots_adjust(top=0.92, hspace=0.43)
    fig.subplots_adjust(hspace=0.43, top=0.835)
    if fig_title:
        fig.suptitle(fig_title, fontsize=20)
    ax = results_df[['loss']].plot(ax=axs[0], color='purple', linewidth=0.7)
    # markers to show distinct points
    results_df[['loss']].plot(ax=axs[0], marker='h', markersize=1, linestyle='none', legend=None)
    results_df[['threshold']].plot(ax=axs[0], color='red', legend=None)

    ax.set_title(f'Reconstruction error of each window (size={window_size}) in compromised program')
    ax.set_xlabel('Index of sliding window')
    ax.set_ylabel('Window reconstruction error')
    ax.legend().remove()

    # threshold = results_df['threshold'].iloc[0]
    last_threshold = results_df['threshold'].iloc[-1]
    # ax.axhline(threshold, color='r', label='Threshold')#, linestyle='--')

    ax_t = ax.twinx()
    ax_t.set_yticks([last_threshold])
    ax_t.set_yticklabels(['Threshold'], fontdict={'fontsize':7})
    ax_t.set_ylim(*ax.get_ylim())
    ax_t.legend().remove()
    # draw line to show where abnormal values actually were
    # ax.axvline(results_df.index.values[4905-window_size], color='k', linestyle='--', label='normal vs abnormal')
    plot_pc_timeline(df_a, ax=axs[1], function_ranges=function_ranges)
    # draw line to show where abnormal values actually were
    # axs[1].axvline(results_df.index.values[4905-window_size], color='k', linestyle='--')
    axs[1].set_title(f'Results ({anomalies_df.shape[0]} anomalous windows were detected)')
    axs[1].set_xlabel('Instruction index')
    axs[1].set_ylabel('Program counter (address)')
    axs[1].get_yaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')

    if not anomalies_df.shape[0]:
        return axs
    # draw anomaly region highlights
    for i, row in anomalies_df.iterrows():
        axs[1].axvspan(row['window_start'], row['window_end'], color='red', alpha=0.15)
    # draw stars
    df_a.iloc[ anomalies_df['window_end'] ].plot(ax=axs[1], color='r', marker='*', markersize=10, linestyle='none', legend=None)
    return axs



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    window_size = 20

    init_settings_i_dont_know()

    normal_f_names = list(glob.glob('../../log_files/*normal*pc'))
    anomaly_f_names = list(glob.glob('../../log_files/*compromised*pc'))

    # use first files only just for simplicity 
    # df_n = df_from_pc_files(normal_f_names).iloc[:,0]
    # df_a = df_from_pc_files(anomaly_f_names).iloc[:,0]

    # each column comes from a different file
    # column name = file name
    # row = program counters from different files
    df_n = df_from_pc_files(normal_f_names) 
    df_a = df_from_pc_files(anomaly_f_names)

    results_df, anomalies_df = detect(df_n, df_a, window_size=window_size)
    plot_results(df_a, results_df, anomalies_df, window_size)
    # plt.legend();
    plt.show()

    # normal_f_names = list(glob.glob('../../log_files/*mimic*pc'))
    # anomaly_f_names = list(glob.glob('../../log_files/*normal.pc'))

