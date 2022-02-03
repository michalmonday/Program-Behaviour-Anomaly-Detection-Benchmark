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

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(sns)

df = pd.read_csv('spx.csv', parse_dates=['date'], index_col='date')

df.head()

plt.plot(df, label='close price')
plt.legend()
plt.show()

train_size = int(len(df) * 0.95)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train[['close']])

train['close'] = scaler.transform(train[['close']])
test['close'] = scaler.transform(test[['close']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 30

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train[['close']], train.close, TIME_STEPS)
X_test, y_test = create_dataset(test[['close']], test.close, TIME_STEPS)

print(X_train.shape)


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
THRESHOLD = 1.7

# results_df = pd.DataFrame(index=test[TIME_STEPS:].index) #(index=test[TIME_STEPS:].index)

results_df = pd.DataFrame(index=df.index.values[TIME_STEPS:-TIME_STEPS])
results_df['loss'] = np.concatenate((train_mae_loss, test_mae_loss))
results_df['threshold'] = THRESHOLD
results_df['anomaly'] = results_df.loss > results_df.threshold
results_df['close'] = test[TIME_STEPS:].close
# results_df[['train_loss', 'test_loss', 'threshold', 'anomaly']].plot()
ax = results_df[['loss']].plot()
ax.axvline(results_df.index.values[train_size], color='k', linestyle='--')
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
  test[TIME_STEPS:].index, 
  scaler.inverse_transform(test[TIME_STEPS:].close.values.reshape(1,-1)).reshape(-1), 
  label='close price'
);

sns.scatterplot(
  anomalies.index,
  scaler.inverse_transform(anomalies.close.values.reshape(1,-1)).reshape(-1),
  color=sns.color_palette()[3],
  s=52,
  label='anomaly'
)
plt.xticks(rotation=25)
plt.legend();
plt.show()

