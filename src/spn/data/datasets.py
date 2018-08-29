'''
Created on March 30, 2018

@author: Alejandro Molina
'''

from os.path import dirname

import numpy as np
import os
from scipy.io import arff
import pandas as pd

path = dirname(__file__) + "/"

def one_hot(y):
  if len(y.shape) != 1:
    return y
  values = np.array(sorted(list(set(y))))
  return np.array([values == v for v in y], dtype=np.int8)

def preproc_data(raw_data):
    train = [list(row) for row in raw_data]

    non_numeric_idx = np.asarray([type(elem) == np.bytes_ for elem in train[0]])
    preproc_train = np.asarray([list(row) for row in train], dtype=np.float)

    non_numeric_train = preproc_train[:, non_numeric_idx]
    numeric_train = preproc_train[:, ~non_numeric_idx]

    # ohe_data_arr = [one_hot(non_numeric_data[:,i]) for i in range(non_numeric_data.shape[1])]
    ohe_data_arr = None
    for i in range(non_numeric_train.shape[1]):
        if ohe_data_arr is None:
            ohe_data_arr = one_hot(non_numeric_train[:, i])
        else:
            ohe_data_arr = np.concatenate((ohe_data_arr, one_hot(non_numeric_train[:, i])), axis=1)

    preproc_data_train = np.concatenate((ohe_data_arr, numeric_train), axis=1)

    return preproc_data_train

def get_nips_data(test_size=0.2):
    fname = path + "count/nips100.csv"
    words = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = np.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    F = len(words)

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(D, test_size=0.2, random_state=42)

    return ("NIPS", np.asarray(words), D, train, test, np.asarray(["discrete"] * F), np.asarray(["poisson"] * F))


def get_binary_data(name):
    train = np.loadtxt(path + "/binary/" + name + ".ts.data", dtype=float, delimiter=",", skiprows=0)
    test = np.loadtxt(path + "/binary/" + name + ".test.data", dtype=float, delimiter=",", skiprows=0)
    valid = np.loadtxt(path + "/binary/" + name + ".valid.data", dtype=float, delimiter=",", skiprows=0)
    D = np.vstack((train, test, valid))
    F = D.shape[1]
    features = ["V" + str(i) for i in range(F)]

    return (
    name.upper(), np.asarray(features), D, train, test, np.asarray(["discrete"] * F), np.asarray(["bernoulli"] * F))


def get_mnist(cachefile=path+'count/mnist.npz'):
    if cachefile and os.path.exists(cachefile):
        npzfile = np.load(cachefile)
        images_tr, labels_tr, images_te, labels_te = npzfile['images_tr'], npzfile['labels_tr'], npzfile['images_te'], npzfile['labels_te']
    else:
        from mnist import MNIST
        mndata = MNIST(path+'count/mnist')
        images_tr, labels_tr = mndata.load_training()
        images_te, labels_te = mndata.load_testing()

        if cachefile:
            np.savez(cachefile, images_tr=images_tr, labels_tr=labels_tr, images_te=images_te, labels_te=labels_te)
    return (images_tr, labels_tr, images_te, labels_te)


def get_categorical_data(name):
    train = arff.loadarff(path + "/categorical/" + name + "/" + name + "-train.arff")
    test = arff.loadarff(path + "/categorical/" + name + "/" + name + "-test.arff")
    valid = arff.loadarff(path + "/categorical/" + name + "/" + name + ".arff")

    train = preproc_data(train[0])
    test = preproc_data(test[0])
    valid = preproc_data(valid[0])

    return (train[:, :-7], train[:, -7:], test[:, :-7], test[:, -7:], valid[:, :-7], valid[:, -7:])

