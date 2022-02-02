#!/usr/bin/python3.7

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import glob
from copy import deepcopy

def read_files(names):
    """ Returns dictionary where
        keys = file names
        values = lists of address integers """
    data = {}
    try: 
        for name in names:
            with open(name) as f:
                short_name = name.split('/')[-1] #name.split('/')[-1].split('.')[0]
                data[short_name] = [int(line.rstrip(), 16) for line in f.readlines()]
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
        
    return data

normal_f_names = list(glob.glob('../log_files/*normal*pc'))
anomaly_f_names = list(glob.glob('../log_files/*compromised*pc'))

# df_n = pd.DataFrame(read_files(normal_f_names))  
# df_a = pd.DataFrame(read_files(anomaly_f_names))  

# use first files only just for simplicity 
npc = pd.DataFrame( list( read_files(normal_f_names).values() )[0] )
apc = pd.DataFrame( list( read_files(anomaly_f_names).values() )[0] )

print(f'normal program run has {len(npc)} program counter values')
print(f'anomalous program run has {len(apc)} program counter values')


#diffs = df.diff()

#df_n.plot(xlabel = 'Index', ylabel='Address')
#plt.show()
#
#df_a.plot(xlabel = 'Index', ylabel='Address')
#plt.show()

# single program is used as normal
# train_data = df_n['trace_0.txt'].values

# this means windows will slide only once
# for the sake of development
# window size will have to be somewhat small I guess 
# (e.g. between 2 - 100 or something, will decide later)
# window_size = train_data.shape[0] - 1 
window_size = 400 #len(npc) - 5

# test_data = train_data.copy()
# MALICIOUS MODIFICATION OF A SINGLE PROGRAM COUNTER VALUE
# test_data[5000] = test_data[5000] + 4 

# train_labels = np.zeros((train_data.shape[0],1))
# test_labels = np.zeros((test_data.shape[0],1))
# test_labels[5000] = 1

# train_data, test_data, train_labels, test_labels = train_test_split(
#     data, labels, test_size=0.2, random_state=21
# )



# this may be done with pandas "rolling" function 
# (probably faster and takes less memory but that's for later)
train_data = npc
test_data = apc

train_data_windows = [ train_data[i: i+window_size] for i in range(train_data.shape[0] - window_size + 1) ]
test_data_windows = [ test_data[i: i+window_size] for i in range(test_data.shape[0] - window_size + 1) ]

train_data_windows = np.array( train_data_windows ).reshape(-1, window_size)
test_data_windows = np.array( test_data_windows ).reshape(-1, window_size)

# remove duplicates
train_data_windows = np.unique(train_data_windows, axis=0)

train_labels = np.zeros((train_data_windows.shape[0],1))
test_labels = np.zeros((test_data_windows.shape[0],1))

min_val = tf.reduce_min(train_data_windows)
max_val = tf.reduce_max(train_data_windows)

# normalization for fast gradient descent
train_data_windows = (train_data_windows - min_val) / (max_val - min_val)
test_data_windows = (test_data_windows - min_val) / (max_val - min_val)

train_data_windows = tf.cast(train_data_windows, tf.float32)
test_data_windows = tf.cast(test_data_windows, tf.float32)



def range_overlaps(r1, r2=(4905, 4917)):
    ''' 4905 - 4917 are indices of program counters in anomalous
        program run where program counters have different values
        from normal run '''
    if range(max(r1[0], r2[0]), min(r1[-1], r2[-1])+1):
        return 1
    return 0

abnormal_range = (4905, 4917)
for i in range(test_data.shape[0] - window_size + 1):
    r1 = (i, i+window_size)
    print(r1, abnormal_range)
    if range_overlaps(r1, abnormal_range):
        test_labels[i] = 1;
        print(f'range overlaps in test_data_windows[{i}]')

abnormal_test_labels_range = (abnormal_range[0] - window_size, abnormal_range[1] - window_size)

# test_labels[5000] = 1



# (x_train, _), (x_test, _) = fashion_mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

# print (x_train.shape)
# print (x_test.shape)

# import pdb; pdb.set_trace()

#(Pdb) p train_data_windows.__len__()
#2
#(Pdb) p train_data_windows[0]
#<tf.Tensor: shape=(9999,), dtype=float32, numpy=
#array([0.23928571, 0.9685714 , 0.9689286 , ..., 0.01571429, 0.01589286,
#       0.01607143], dtype=float32)>
#(Pdb) p train_data_windows[1]
#<tf.Tensor: shape=(9999,), dtype=float32, numpy=
#array([0.9685714 , 0.9689286 , 0.9692857 , ..., 0.01589286, 0.01607143,
#       0.01625   ], dtype=float32)>
# test_data_windows = [ test_data[i: i+window_size] for i in range(test_data.shape[0] - window_size + 1) ]

class AnomalyDetector(Model):
  def __init__(self, latent_dim):
    super(AnomalyDetector, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(window_size, activation="sigmoid")])
    # self.encoder = tf.keras.Sequential([
    #   layers.Dense(64, activation="relu"),
    #   layers.Dense(32, activation="relu"),
    #   layers.Dense(32, activation="relu"),
    #   layers.Dense(16, activation="relu"),
    # 2D array, got 1D arra   layers.Dense(8, activation="relu")])

    # self.decoder = tf.keras.Sequential([
    #   layers.Dense(16, activation="relu"),
    #   layers.Dense(32, activation="relu"),
    #   layers.Dense(32, activation="relu"),
    #   layers.Dense(64, activation="relu"),
    #   layers.Dense(window_size, activation="sigmoid")])

    # self.encoder = tf.keras.Sequential([
    #   layers.Dense(128, activation="relu"),
    #   layers.Dense(64, activation="relu"),
    #   layers.Dense(32, activation="relu"),
    #   layers.Dense(16, activation="relu"),
    #   layers.Dense(8, activation="relu")])

    # self.decoder = tf.keras.Sequential([
    #   layers.Dense(16, activation="relu"),
    #   layers.Dense(32, activation="relu"),
    #   layers.Dense(64, activation="relu"),
    #   layers.Dense(128, activation="relu"),
    #   layers.Dense(10000, activation="sigmoid")])


  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector(window_size)
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(train_data_windows, train_data_windows, 
          epochs=200,
          batch_size = 5000,
          validation_data=(train_data_windows, train_data_windows), # overfitting is the whole point when using autoencoder for anomaly detection
          shuffle=True)

#plt.plot(history.history["loss"], label="Training Loss")
#plt.plot(history.history["val_loss"], label="Validation Loss")
#plt.legend()
#plt.show()


def plot_reconstruction(data_windows, index):
    encoded_data = autoencoder.encoder(data_windows).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
    plt.plot( data_windows[index], 'b' )
    plt.plot( decoded_data[index], 'r' )
    plt.fill_between(
            np.arange(data_windows.shape[1]),
            decoded_data[index],
            data_windows[index],
            color='lightcoral'
            )
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

# plot_reconstruction(test_data_windows, 4912)
# plot_reconstruction(train_data_windows, train_data_windows.shape[0] - test_data_windows.shape[0] + 4912)

reconstructions = autoencoder.predict(train_data_windows)
train_loss = tf.keras.losses.mae(reconstructions, train_data_windows)
# threshold = np.mean(train_loss) + np.std(train_loss)
# print('Automatic threshold = {threshold}')
threshold = 0.168

reconstructions = autoencoder.predict(test_data_windows)
test_loss = tf.keras.losses.mae(reconstructions, test_data_windows)
abnormal_mean_loss = np.mean( test_loss[ abnormal_test_labels_range[0] : abnormal_test_labels_range[1] ] )
train_mean_loss = np.mean(train_loss)
print(f'abnormal_mean_loss = {abnormal_mean_loss}, train_mean_loss = {train_mean_loss}')

plt.plot(test_loss.numpy())
plt.plot(train_loss.numpy())
plt.show()
plt.hist(test_loss.numpy(), alpha=0.3, bins=50)
plt.hist(train_loss.numpy(), alpha=0.3, bins=50)
plt.show()

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    labels = labels.reshape(-1) == 1.
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

preds = predict(autoencoder, test_data_windows, threshold)
print_stats(preds, test_labels)

# preds = predict(autoencoder, train_data, threshold)
# print_stats(preds, train_labels)



# encoded_data = autoencoder.encoder(normal_test_data).numpy()
# decoded_data = autoencoder.decoder(encoded_data).numpy()
# 
# plt.plot(normal_test_data[0], 'b')
# plt.plot(decoded_data[0], 'r')
# plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
# plt.legend(labels=["Input", "Reconstruction", "Error"])
# plt.show()



# (x_train, _), (x_test, _) = fashion_mnist.load_data()
# 
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# 
# print (x_train.shape)
# print (x_test.shape)
# 
# latent_dim = 64 
# 
# class Autoencoder(Model):
#   def __init__(self, latent_dim):
#     super(Autoencoder, self).__init__()
#     self.latent_dim = latent_dim   
#     self.encoder = tf.keras.Sequential([
#       layers.Flatten(),
#       layers.Dense(latent_dim, activation='relu'),
#     ])
#     self.decoder = tf.keras.Sequential([
#       layers.Dense(784, activation='sigmoid'),
#       layers.Reshape((28, 28))
#     ])
# 
#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded
# 
# autoencoder = Autoencoder(latent_dim)
# 
# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
# 
# autoencoder.fit(x_train, x_train,
#                 epochs=10,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
# 
# encoded_imgs = autoencoder.encoder(x_test).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
# 
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#   # display original
#   ax = plt.subplot(2, n, i + 1)
#   plt.imshow(x_test[i])
#   plt.title("original")
#   plt.gray()
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
# 
#   # display reconstruction
#   ax = plt.subplot(2, n, i + 1 + n)
#   plt.imshow(decoded_imgs[i])
#   plt.title("reconstructed")
#   plt.gray()
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
# plt.show()
