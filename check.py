import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
from os.path import isfile, isdir

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

count_right = 0
count_wrong = 0
count_total = 0

test = pd.read_csv("output.csv")["Label"]

for i in range(0,len(y_test)):
    if test[i] == y_test[i]:
        count_right += 1
    else:
        count_wrong += 1
    count_total += 1

print("Correct:", count_right)
print("Wrong:", count_wrong)
print("Ratio:", count_right/count_total)
