#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
from os.path import isfile, isdir
from sklearn.model_selection import train_test_split

def input_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    # (x_train, y_train), (x_test, y_test)
    # let's print the shape before we reshape and normalize
    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)
    print("x_test shape", x_test.shape)
    print("y_test shape", y_test.shape)

    # building the input vector from the 28x28 pixels
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # print the final input shape ready for training
    print("Train matrix shape", x_train.shape)
    print("Test matrix shape", x_test.shape)

    train = pd.DataFrame(y_train, columns= ["label"]).join(pd.DataFrame(x_train))
    test = pd.DataFrame(y_test, columns= ["label"]).join(pd.DataFrame(x_test))

    # check training data
    trainLabelCounts = train['label'].value_counts(sort = False)

def input(train_path, test_path):
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)

    print(train.shape)
    print(test.shape)


# input('data/emnist-digits-train.csv','data/emnist-digits-test.csv')

#%%
train_path = 'data/emnist-digits-train.csv'
test_path = 'data/emnist-digits-test.csv'

train = pd.read_csv(train_path, header=None)
test = pd.read_csv(test_path, header=None)

x_train = train[train.columns[1:]].copy()
y_train = train[train.columns[0:1]].copy()

x_test = test[test.columns[1:]].copy()
y_test = test[test.columns[0:1]].copy()

print("Train shape:", x_train.shape, y_train.shape)

print("Test shape:", x_test.shape, y_test.shape)

columns = ["label"] + ["pixel"] * (len(train.columns) - 1)

train.columns = columns
test.columns = columns



