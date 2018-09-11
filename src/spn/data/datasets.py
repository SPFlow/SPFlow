'''
Created on March 30, 2018

@author: Alejandro Molina
'''

from os.path import dirname

import numpy as np
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
from spn.structure.leaves.histogram.Histograms import Histogram


path = dirname(__file__) + "/"


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


def load_from_csv(data_file, header=0, histogram=True):
    df = pd.read_csv(data_file, delimiter=",", header=header)
    df = df.dropna(axis=0, how='any')

    feature_names = df.columns.values.tolist() if header == 0 else [
        "X_{}".format(i) for i in range(len(df.columns))]

    dtypes = df.dtypes
    feature_types = []
    for feature_type in dtypes:
        if feature_type.kind == 'O':
            feature_types.append(Histogram)
        else:
            feature_types.append(PiecewiseLinear)

    data_dictionary = {
        'features': [{"name": name,
                      "type": typ,
                      "pandas_type": dtypes[i]}
                     for i, (name, typ)
                     in enumerate(zip(feature_names, feature_types))],
        'num_entries': len(df)
    }

    idx = df.columns

    for id, name in enumerate(idx):
        if feature_types[id] == Histogram:
            lb = LabelEncoder()
            data_dictionary['features'][id]["encoder"] = lb
            df[name] = df[name].astype('category')
            df[name] = lb.fit_transform(df[name])
            data_dictionary['features'][id]["values"] = lb.transform(
                lb.classes_)
        if dtypes[id].kind == 'M':
            df[name] = (df[name] - df[name].min()) / np.timedelta64(1, 'D')

    data = np.array(df)

    return data, feature_types, data_dictionary
