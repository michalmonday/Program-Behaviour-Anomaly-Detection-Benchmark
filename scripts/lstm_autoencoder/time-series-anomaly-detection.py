#!/usr/bin/env python
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

from utils import read_pc_values, df_from_pc_files

def init_settings_i_dont_know():
    register_matplotlib_converters()
    # sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    # rcParams['figure.figsize'] = 22, 10
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

init_settings_i_dont_know()


normal_f_names = list(glob.glob('../../log_files/*normal*pc'))
anomaly_f_names = list(glob.glob('../../log_files/*compromised*pc'))

# use first files only just for simplicity 
# npc = df_from_pc_files(normal_f_names).iloc[:,0]
# apc = df_from_pc_files(anomaly_f_names).iloc[:,0]

# each column comes from a different file
# column name = file name
# row = program counters from different files
npc = df_from_pc_files(normal_f_names) 
apc = df_from_pc_files(anomaly_f_names)


print(sns)

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

window_size = 20

train_data = npc
test_data = apc

# for training data duplicate windows are dropped
# it greatly improves training times
X_train = utils.multiple_files_df_program_counters_to_sliding_windows(npc, window_size).drop_duplicates().values

# for testing data speed isn't a problem (predictions are done relatively fast)
# so duplicates don't have to be dropped (which is good because it wouldn't be good for presenting results)
X_test = utils.multiple_files_df_program_counters_to_sliding_windows(apc, window_size).values

min_val = tf.reduce_min(X_train)
max_val = tf.reduce_max(X_train)

# normalization for fast gradient descent
X_train = (X_train - min_val) / (max_val - min_val)
X_test = (X_test - min_val) / (max_val - min_val)

X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)

def produce_y(X_windows):
    ''' y is the first program counter value following the last 
        program counter of a window '''
    ys = []
    for i, window in enumerate(X_windows):
        if i == 0:
            continue
        ys.append(window[-1]) 
    return np.array(ys)

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

# results_df = pd.DataFrame( index=test_data.index.values[window_size:-window_size] )
results_df = pd.DataFrame( index=test_data.index.values[:-window_size] )
# results_df['loss'] = np.concatenate((train_mae_loss, test_mae_loss))
results_df['loss'] = test_mae_loss
results_df['threshold'] = THRESHOLD
results_df['anomaly'] = results_df.loss > results_df.threshold
# results_df['close'] = test[window_size:].close
# results_df[['train_loss', 'test_loss', 'threshold', 'anomaly']].plot()

ax = results_df[['loss']].plot()
ax.set_xlabel('Index of sliding window')
ax.set_ylabel('Window reconstruction error (big error = probably anomaly)')
ax.axvline(results_df.index.values[4905-window_size], color='k', linestyle='--')
plt.show()

#plt.plot(results_df.index, results_df.loss, label='loss')
#plt.plot(results_df.index, results_df.threshold, label='threshold')
#plt.xticks(rotation=25)
#plt.legend()
#plt.show()

anomalies = results_df[results_df.anomaly == True]
anomalies.head()

plt.plot(
  test_data[window_size:].index, 
  test_data[window_size:].values
  # label=''
)
ax = plt.gca()
ax.axvline(results_df.index.values[4905], color='k', linestyle='--')
ax.set_xlabel('Instruction index')
ax.set_ylabel('Program counter value (address)')
ax.get_yaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')


sns.scatterplot(
  anomalies.index + window_size,
  test_data.iloc[anomalies.index + window_size].values.reshape(-1),
  color=sns.color_palette()[3],
  s=52,
  label='anomaly'
)
plt.xticks(rotation=25)
plt.legend();
plt.show()

# plt.plot(
#   test_data[window_size:].index, 
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
