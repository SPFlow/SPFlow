'''
Created on March 30, 2018

@author: Alejandro Molina
'''

from os.path import dirname

import numpy as np
import os
import arff
from scipy.io.arff import loadarff
import pandas as pd
import xml.etree.ElementTree as ET

path = dirname(__file__) + "/"

def one_hot(y, values):
  if len(y.shape) != 1:
    return y
  values = np.array(sorted(list(set(values))))
  return np.array([values == v for v in y], dtype=np.int8)


def transpose_list(data):
    return list(map(list, zip(*data)))


def preproc_arff_data(raw_data, labels):

    data = raw_data['data']
    data_transposed = transpose_list(data)
    labels = [child.attrib['name'] for child in labels.getroot()]
    labels_idx = np.asarray([elem[0] in labels for elem in raw_data['attributes']])
    numeric_idx = np.asarray([elem[1] == 'NUMERIC'for elem in raw_data['attributes']])
    values = [elem[1] for elem in raw_data['attributes']]   # the range of ohe

    num_data_rows = len(data)
    num_labels = len(labels)
    num_data_cols = len(raw_data['attributes'])
    num_input_cols = num_data_cols - num_labels


    # split input and labels
    input_transposed = np.asarray([input for i, input in enumerate(data_transposed) if labels_idx[i] == False])
    values_input = [value for i, value in enumerate(values) if labels_idx[i] == False]
    labels = [one_hot(np.asarray(label), values[i]) for i, label in enumerate(data_transposed) if labels_idx[i] == True]   # do we need to ohe labels?
    labels_ohe = np.swapaxes(np.asarray(labels), 0, 1).reshape(num_data_rows, -1)   # shape is now (#instance, #labels, #ohe)

    ohe_data_arr = None
    for i in range(num_input_cols):
        if ohe_data_arr is None:
            if numeric_idx[i] == False:
                ohe_data_arr = one_hot(input_transposed[i], values[i]).reshape(-1,num_data_rows)
            else:
                ohe_data_arr = input_transposed[i].reshape(-1,num_data_rows)
        else:
            if numeric_idx[i] == False:
                ohe_data_arr = np.concatenate((ohe_data_arr, one_hot(input_transposed[i], values[i]).reshape(-1,num_data_rows)), axis=0)
            else:
                ohe_data_arr = np.concatenate((ohe_data_arr, input_transposed[i].reshape(-1, num_data_rows)), axis=0)

    return transpose_list(ohe_data_arr), labels_ohe

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

    cachefile = path + 'count/' + name + '.npz'

    if cachefile and os.path.exists(cachefile):
        npzfile = np.load(cachefile)
        train_input, train_labels, test_input, test_labels = npzfile['train_input'], npzfile['train_labels'], npzfile['test_input'], npzfile['test_labels']
    else:
        train = arff.load(open(path + "/categorical/" + name + "/" + name + "-train.arff", 'r'))
        test = arff.load(open(path + "/categorical/" + name + "/" + name + "-test.arff", 'r'))
        labels = ET.parse(path + "/categorical/" + name + "/" + name + '.xml')

        train_input, train_labels = preproc_arff_data(train, labels)
        test_input, test_labels = preproc_arff_data(test, labels)

        if cachefile:
            np.savez(cachefile, train_input=train_input, train_labels=train_labels, test_input=test_input, test_labels=test_labels)

    return (train_input, train_labels, test_input, test_labels)

