'''
Created on March 24, 2018

@author: Alejandro Molina
'''
import codecs

import numpy as np
from joblib import Memory
from sklearn.model_selection import train_test_split

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.StructureLearning import learn_structure, next_operation
from spn.algorithms.splitting.KMeans import split_rows_KMeans
from spn.algorithms.splitting.RDC import split_cols_RDC, split_rows_RDC
from spn.io.Text import to_JSON, to_str_equation
from spn.leaves.Histograms import add_domains, create_histogram_leaf, Histogram_to_str_equation, Histogram_Likelihoods
from spn.structure.Base import *

memory = Memory(cachedir="cache", verbose=0, compress=9)


#@memory.cache
def learn(data, ds_context):
    split_cols = lambda data, ds_context, scope: split_cols_RDC(data, ds_context, scope, threshold=0.3, linear=False)
    nextop = lambda data, no_clusters=False, no_independencies=False, is_first=False, cluster_first=True,\
                    cluster_univariate=False, min_instances_slice=200: next_operation(data, no_clusters,
                                                                                      no_independencies,
                                                                                      is_first,
                                                                                      cluster_first,
                                                                                      cluster_univariate,
                                                                                      min_instances_slice)


    split_rows = lambda data, ds_context, scope: split_rows_KMeans(data, ds_context, scope, pre_proc="log+1")
    spn = learn_structure(data, ds_context, split_rows, split_cols, create_histogram_leaf, nextop)

    return spn


def get_nips_data():
    fname = "data/nips100.csv"
    words = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = np.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    return ("NIPS", D, np.asarray(words))


# def trainSPN(outprefix, data, words, top_n_features):


if __name__ == '__main__':
    ds_name, data, words = get_nips_data()

    top_n_features = 40

    train, test = train_test_split(data[:, 0:top_n_features], test_size=0.2, random_state=42)

    ds_context = Context()
    ds_context.statistical_type = np.asarray(["discrete"] * top_n_features)
    ds_context.distribution_family = np.asarray(["poisson"] * top_n_features)

    add_domains(train, ds_context)
    spn = learn(train, ds_context)

    print(get_structure_stats(spn))

    outprefix = "spns/%s_%s/" % (ds_name, top_n_features)

    with open(outprefix + "eqq.txt", "w") as text_file:
        print(to_str_equation(spn, Histogram_to_str_equation, words[0:top_n_features]), file=text_file)

    with codecs.open(outprefix + "spn.json", "w", "utf-8-sig") as text_file:
        text_file.write(to_JSON(spn))

    np.savetxt(outprefix + "traindata.txt", train, delimiter=";", header=";".join(words))

    np.savetxt(outprefix + "testdata.txt", test, delimiter=";", header=";".join(words))
    testll = log_likelihood(spn, test, Histogram_Likelihoods)
    np.savetxt(outprefix + "ll.txt", testll)

    print(np.mean(testll))
