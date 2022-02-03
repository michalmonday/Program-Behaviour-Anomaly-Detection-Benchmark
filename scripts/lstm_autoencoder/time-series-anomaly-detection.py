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

normal_f_names = list(glob.glob('../../log_files/*normal*pc'))
anomaly_f_names = list(glob.glob('../../log_files/*compromised*pc'))

# use first files only just for simplicity 
npc = df_from_pc_files(normal_f_names).iloc[:,0]
apc = df_from_pc_files(anomaly_f_names).iloc[:,0]

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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

train_data_windows = [ train_data[i: i+window_size] for i in range(train_data.shape[0] - window_size + 1) ]
test_data_windows = [ test_data[i: i+window_size] for i in range(test_data.shape[0] - window_size + 1) ]

train_data_windows = np.array( train_data_windows ).reshape(-1, window_size)
test_data_windows = np.array( test_data_windows ).reshape(-1, window_size)

# remove duplicates
# train_data_windows = np.unique(train_data_windows, axis=0)

train_labels = np.zeros((train_data_windows.shape[0],1))
test_labels = np.zeros((test_data_windows.shape[0],1))

min_val = tf.reduce_min(train_data_windows)
max_val = tf.reduce_max(train_data_windows)

# normalization for fast gradient descent
train_data_windows = (train_data_windows - min_val) / (max_val - min_val)
test_data_windows = (test_data_windows - min_val) / (max_val - min_val)

train_data_windows = tf.cast(train_data_windows, tf.float32)
test_data_windows = tf.cast(test_data_windows, tf.float32)

# X_train = train_data_windows.numpy().reshape(-1, window_size, 1)

# def create_dataset(X, y, window_size=1):
#     Xs, ys = [], []
#     for i in range(len(X) - window_size):
#         v = X.iloc[i:(i + window_size)].values
#         Xs.append(v)        
#         ys.append(y.iloc[i + window_size])
#     return np.array(Xs), np.array(ys)

def produce_y(X_windows):
    ys = []
    for i, window in enumerate(X_windows):
        if i == 0:
            continue
        ys.append(window[-1]) 
    return np.array(ys)

# reshape to [samples, window_size, n_features]

# X_train, y_train = create_dataset(train[['close']], train.close, window_size)
# X_test, y_test = create_dataset(test[['close']], test.close, window_size)

y_train = produce_y( train_data_windows )
X_train = np.array( train_data_windows[:-1] ).reshape(-1, window_size, 1)

y_test = produce_y( test_data_windows )
X_test = np.array( test_data_windows[:-1] ).reshape(-1, window_size, 1)

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
    epochs=50,
    batch_size=5000,#32,
    validation_split=0.1,
    shuffle=False
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();
plt.show()

X_train_pred = model.predict(X_train)

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

sns.distplot(train_mae_loss, bins=50, kde=True);

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
ax.axvline(results_df.index.values[4905-window_size], color='k', linestyle='--')
# plt.plot(train_mae_loss, label='train_loss')
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
);

sns.scatterplot(
  anomalies.index + window_size,
  # anomalies.loss.values,
  test_data[anomalies.index + window_size].values,
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
