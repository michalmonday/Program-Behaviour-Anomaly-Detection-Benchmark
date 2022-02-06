#!/usr/bin/python3.7
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import glob

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import utils
from copy import deepcopy
from utils import read_pc_values, df_from_pc_files, plot_pc_timeline

def init_settings_i_dont_know():
    register_matplotlib_converters()
    # sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    # rcParams['figure.figsize'] = 22, 10
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

def produce_y(X_windows):
    ''' y is the first program counter value following the last 
        program counter of a window '''
    ys = []
    for i, window in enumerate(X_windows):
        if i == 0:
            continue
        ys.append(window[-1]) 
    return np.array(ys)

def detect_by_lstm_autoencoder(df_n, df_a, window_size=20):
    # for training data duplicate windows are dropped
    # it greatly improves training times
    X_train = utils.multiple_files_df_program_counters_to_sliding_windows(df_n, window_size).drop_duplicates().values

    # for testing data speed isn't a problem (predictions are done relatively fast)
    # so duplicates don't have to be dropped (which is good because it wouldn't be good for presenting results)
    X_test = utils.multiple_files_df_program_counters_to_sliding_windows(df_a, window_size).values

    min_val = tf.reduce_min(X_train)
    max_val = tf.reduce_max(X_train)

    # normalization for fast gradient descent
    X_train = (X_train - min_val) / (max_val - min_val)
    X_test = (X_test - min_val) / (max_val - min_val)

    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)

# reshape to [samples, window_size, n_features]

# X_train, y_train = create_dataset(train[['close']], train.close, window_size)
# X_test, y_test = create_dataset(test[['close']], test.close, window_size)

    y_train = produce_y( X_train )
    X_train = np.array( X_train[:-1] ).reshape(-1, window_size, 1)

    y_test = produce_y( X_test )
    X_test = np.array( X_test[:-1] ).reshape(-1, window_size, 1)

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

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=5000,#32,
        validation_split=0.1,
        shuffle=False
    )

    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend();
    # plt.show()

    X_train_pred = model.predict(X_train)

    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    # sns.distplot(train_mae_loss, bins=50, kde=True);

    X_test_pred = model.predict(X_test)

    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    # THRESHOLD = 0.65
    THRESHOLD = train_mae_loss.max()

    # results_df = pd.DataFrame(index=test[window_size:].index) #(index=test[window_size:].index)

    # results_df = pd.DataFrame( index=df_a.index.values[window_size:-window_size] )
    results_df = pd.DataFrame( index=df_a.index.values[:-window_size] )
    # results_df['loss'] = np.concatenate((train_mae_loss, test_mae_loss))
    results_df['loss'] = test_mae_loss
    results_df['threshold'] = THRESHOLD
    results_df['anomaly'] = results_df.loss > results_df.threshold
    results_df['window_start'] = results_df.index.values
    results_df['window_end'] = results_df['window_start'] + window_size

    # results_df['close'] = test[window_size:].close
    # results_df[['train_loss', 'test_loss', 'threshold', 'anomaly']].plot()


    #plt.plot(results_df.index, results_df.loss, label='loss')
    #plt.plot(results_df.index, results_df.threshold, label='threshold')
    #plt.xticks(rotation=25)
    #plt.legend()
    #plt.show()

    anomalies = results_df[results_df.anomaly == True]
    return results_df, anomalies



if __name__ == '__main__':
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

    results_df, anomalies = detect_by_lstm_autoencoder(df_n, df_a, window_size=window_size)
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(top=0.92, hspace=0.43)
    ax = results_df[['loss']].plot(ax=axs[0], color='purple', linewidth=0.7)
    # markers to show distinct points
    results_df[['loss']].plot(ax=axs[0], marker='h', markersize=1, linestyle='none', legend=None)

    ax.set_title('Reconstruction error of each window in compromised program')
    ax.set_xlabel('Index of sliding window')
    ax.set_ylabel('Window reconstruction error')
    ax.legend().remove()

    # ax.axvline(results_df.index.values[4905-window_size], color='k', linestyle='--')

    threshold = results_df['threshold'].iloc[0]
    ax.axhline(threshold, color='r', label='Threshold')#, linestyle='--')
    ax_t = ax.twinx()
    ax_t.set_yticks([threshold])
    ax_t.set_yticklabels(['Threshold'], fontdict={'fontsize':7})
    ax_t.set_ylim(*ax.get_ylim())
    ax_t.legend().remove()

    ax.axvline(results_df.index.values[4905-window_size], color='k', linestyle='--', label='normal vs abnormal')

    # axs[1].plot(
    #   df_a[window_size:].index, 
    #   df_a[window_size:].values
    # )

    # df_a[window_size:].plot(ax=axs[1])
    plot_pc_timeline(df_a, ax=axs[1])

    # draw line to show where abnormal values actually were
    axs[1].axvline(results_df.index.values[4905], color='k', linestyle='--')
    axs[1].set_title('Results (anomalies are marked on PC timeline if found)')
    axs[1].set_xlabel('Instruction index')
    axs[1].set_ylabel('Program counter (address)')
    axs[1].get_yaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')

    # draw anomaly region highlights
    for i, row in anomalies.iterrows():
        axs[1].axvspan(row['window_start'], row['window_end'], color='red', alpha=0.15)

    # draw stars
    df_a.iloc[ anomalies['window_end'] ].plot(ax=axs[1], color='r', marker='*', markersize=10, linestyle='none', legend=None)

    # import pdb; pdb.set_trace()
    # anomalies.plot( ax=axs[1])

    #sns.scatterplot( anomalies.index + window_size,
    #  df_a.iloc[anomalies.index + window_size].values.reshape(-1),
    #  color=sns.color_palette()[3],
    #  s=52,
    #  label='anomaly'
    #)
    plt.xticks(rotation=25)
    plt.legend();
    plt.show()

    # normal_f_names = list(glob.glob('../../log_files/*mimic*pc'))
    # anomaly_f_names = list(glob.glob('../../log_files/*normal.pc'))



# df = pd.read_csv('spx.csv', parse_dates=['date'], index_col='date')
# df.head()
# plt.plot(df, label='close price')
# plt.legend()
# plt.show()
# train_size = int(len(df) * 0.95)
# test_size = len(df) - train_size
# train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
# print(train.shape, test.shape)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler = scaler.fit(train[['close']])
# train['close'] = scaler.transform(train[['close']])
# test['close'] = scaler.transform(test[['close']])

# plt.plot(
#   df_a[window_size:].index, 
#   scaler.inverse_transform(test[window_size:].close.values.reshape(1,-1)).reshape(-1), 
#   label='close price'
# );

# sns.scatterplot(
#   anomalies.index,
#   scaler.inverse_transform(anomalies.close.values.reshape(1,-1)).reshape(-1),
#   color=sns.color_palette()[3],
#   s=52,
#   label='anomaly'
# )
# plt.xticks(rotation=25)
# plt.legend();
# plt.show()
