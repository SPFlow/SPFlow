'''
Created on March 27, 2018

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
from spn.algorithms.splitting.RDC import split_cols_RDC
from spn.io.Text import to_JSON, to_str_equation, str_to_spn, to_str_ref_graph
from spn.leaves.Histograms import add_domains, create_histogram_leaf, Histogram_to_str_equation, Histogram_Likelihoods, \
    Histogram_str_to_spn
import os

from spn.structure.Base import Context

np.set_printoptions(precision=30)

memory = Memory(cachedir="cache", verbose=0, compress=9)


#@memory.cache
def learn(data, ds_context, linear=False):
    split_cols = lambda data, ds_context, scope: split_cols_RDC(data, ds_context, scope, threshold=0.00000001, linear=linear)
    nextop = lambda data, no_clusters=False, no_independencies=False, is_first=False, cluster_first=True, \
                    cluster_univariate=False, min_instances_slice=200: next_operation(data, no_clusters,
                                                                                      no_independencies,
                                                                                      is_first,
                                                                                      cluster_first,
                                                                                      cluster_univariate,
                                                                                      min_instances_slice)

    spn = learn_structure(data, ds_context, split_rows_KMeans, split_cols, create_histogram_leaf, nextop)

    return spn


def get_RL_data():
    path = os.path.dirname(__file__)
    fname = path + "/frozen_lake.csv"
    words = ["state", "action", "next_state"]
    D = np.loadtxt(fname, dtype=float, delimiter=",", skiprows=0)
    F = len(words)
    train, test = train_test_split(D, test_size=0.2, random_state=42)

    return ("FROZEN_LAKE", D, train, test, np.asarray(words), np.asarray(["discrete"] * F), np.asarray(["categorical"] * F))



if __name__ == '__main__':
    ds_name, data, train, test, words, statistical_type, distribution_family = get_RL_data()

    ds_context = Context()
    ds_context.statistical_type = statistical_type
    ds_context.distribution_family = distribution_family
    add_domains(data, ds_context)

    spn = learn(train, ds_context, True)

    print(get_structure_stats(spn))



