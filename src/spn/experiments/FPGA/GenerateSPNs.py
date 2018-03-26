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
from spn.algorithms.StructureLearning import learn_structure, next_operation, Context
from spn.algorithms.splitting.KMeans import split_rows_KMeans
from spn.algorithms.splitting.RDC import split_cols_RDC
from spn.io.Text import to_JSON, to_str_equation, str_to_spn
from spn.leaves.Histograms import add_domains, create_histogram_leaf, Histogram_to_str_equation, Histogram_Likelihoods, \
    Histogram_str_to_spn
import os


np.set_printoptions(precision=30)

memory = Memory(cachedir="cache", verbose=0, compress=9)


@memory.cache
def learn(data, ds_context):
    split_cols = lambda data, ds_context, scope: split_cols_RDC(data, ds_context, scope, threshold=0.3, linear=False)
    nextop = lambda data, no_clusters=False, no_independencies=False, is_first=False, cluster_first=True, \
                    cluster_univariate=False, min_instances_slice=200: next_operation(data, no_clusters,
                                                                                      no_independencies,
                                                                                      is_first,
                                                                                      cluster_first,
                                                                                      cluster_univariate,
                                                                                      min_instances_slice)

    spn = learn_structure(data, ds_context, split_rows_KMeans, split_cols, create_histogram_leaf, nextop)

    return spn


def get_nips_data():
    path = os.path.dirname(__file__)
    fname = path + "/data/nips100.csv"
    words = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = np.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    F = len(words)
    return ("NIPS", D, np.asarray(words), np.asarray(["discrete"] * F), np.asarray(["poisson"] * F))


def run_experiment(dataset, top_n_features):
    ds_name, data, words, statistical_type, distribution_family = dataset

    data = data[:, 0:top_n_features]
    words = words[0:top_n_features]

    ds_context = Context()
    ds_context.statistical_type = statistical_type
    ds_context.distribution_family = distribution_family
    add_domains(data, ds_context)

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    spn = learn(train, ds_context)

    print(get_structure_stats(spn))

    path = os.path.dirname(__file__)
    outprefix = path + "/spns/%s_%s/" % (ds_name, top_n_features)

    if not os.path.exists(outprefix):
        os.makedirs(outprefix)

    with open(outprefix + "eqq.txt", "w") as text_file:
        print(to_str_equation(spn, Histogram_to_str_equation, words), file=text_file)

    with codecs.open(outprefix + "spn.json", "w", "utf-8-sig") as text_file:
        text_file.write(to_JSON(spn))

    np.savetxt(outprefix + "all_data.txt", data, delimiter=";", header=";".join(words))
    np.savetxt(outprefix + "train_data.txt", train, delimiter=";", header=";".join(words))
    np.savetxt(outprefix + "test_data.txt", test, delimiter=";", header=";".join(words))

    np.savetxt(outprefix + "all_data_ll.txt", log_likelihood(spn, data, Histogram_Likelihoods))
    np.savetxt(outprefix + "test_ll.txt", log_likelihood(spn, test, Histogram_Likelihoods))
    np.savetxt(outprefix + "train_ll.txt", log_likelihood(spn, train, Histogram_Likelihoods))


def load_spn_from_file(outprefix):
    with open(outprefix+'eqq.txt', 'r') as myfile:
        eq = myfile.read()
    with open(outprefix+'all_data.txt', 'r') as myfile:
        words = myfile.readline().strip()
        words = words[2:]
        words = words.split(';')

    spn = str_to_spn(eq, words, Histogram_str_to_spn)
    return spn, words, eq

if __name__ == '__main__':

    ds_name, data, words, statistical_type, distribution_family = get_nips_data()

    for topn in [5,10,20,30,40,50,60,70,80]:
        run_experiment(get_nips_data(), topn)



