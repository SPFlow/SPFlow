'''
Created on March 24, 2018

@author: Alejandro Molina
'''
import codecs
import os

import numpy as np
from joblib import Memory
from sklearn.model_selection import train_test_split

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.StructureLearning import learn_structure, next_operation
from spn.algorithms.splitting.KMeans import split_rows_KMeans
from spn.algorithms.splitting.RDC import split_cols_RDC
from spn.io.Text import to_JSON, to_str_equation, str_to_spn, to_str_ref_graph
from spn.leaves.Histograms import add_domains, create_histogram_leaf, Histogram_str_to_spn, histogram_to_str, \
    histogram_likelihood
from spn.structure.Base import Context

np.set_printoptions(precision=30)

memory = Memory(cachedir="cache", verbose=0, compress=9)


@memory.cache
def learn(data, ds_context, linear=False):
    split_cols = lambda data, ds_context, scope: split_cols_RDC(data, ds_context, scope, threshold=0.3, linear=linear)
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
    train, test = train_test_split(D, test_size=0.2, random_state=42)

    return ("NIPS", D, train, test, np.asarray(words), np.asarray(["discrete"] * F), np.asarray(["poisson"] * F))


def get_binary_data(name):
    path = os.path.dirname(__file__)
    train = np.loadtxt(path + "/data/binary/" + name + ".ts.data", dtype=float, delimiter=",", skiprows=0)
    test = np.loadtxt(path + "/data/binary/" + name + ".test.data", dtype=float, delimiter=",", skiprows=0)
    valid = np.loadtxt(path + "/data/binary/" + name + ".valid.data", dtype=float, delimiter=",", skiprows=0)
    D = np.vstack((train,test, valid))
    F = D.shape[1]
    words = ["V"+str(i) for i in range(F)]

    return (name.upper(), D, train, test, np.asarray(words), np.asarray(["discrete"] * F), np.asarray(["bernoulli"] * F))


def run_experiment(dataset, top_n_features, linear=False):
    ds_name, data, train, test, words, statistical_type, distribution_family = dataset

    data = data[:, 0:top_n_features]
    words = words[0:top_n_features]
    train = train[:, 0:top_n_features]
    test = test[:, 0:top_n_features]

    ds_context = Context()
    ds_context.statistical_type = statistical_type
    ds_context.distribution_family = distribution_family
    add_domains(data, ds_context)


    spn = learn(train, ds_context, linear)

    print(get_structure_stats(spn))

    path = os.path.dirname(__file__)
    outprefix = path + "/spns/%s_%s/" % (ds_name, top_n_features)

    if not os.path.exists(outprefix):
        os.makedirs(outprefix)

    with open(outprefix + "eqq.txt", "w") as text_file:
        print(to_str_equation(spn, histogram_to_str, words), file=text_file)

    with open(outprefix + "spn.txt", "w") as text_file:
        print(to_str_ref_graph(spn, histogram_to_str, words), file=text_file)

    with codecs.open(outprefix + "spn.json", "w", "utf-8-sig") as text_file:
        text_file.write(to_JSON(spn))

    np.savetxt(outprefix + "all_data.txt", data, delimiter=";", header=";".join(words))
    np.savetxt(outprefix + "train_data.txt", train, delimiter=";", header=";".join(words))
    np.savetxt(outprefix + "test_data.txt", test, delimiter=";", header=";".join(words))

    np.savetxt(outprefix + "all_data_ll.txt", log_likelihood(spn, data, histogram_likelihood))
    np.savetxt(outprefix + "test_ll.txt", log_likelihood(spn, test, histogram_likelihood))
    np.savetxt(outprefix + "train_ll.txt", log_likelihood(spn, train, histogram_likelihood))


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


    # for bname in ["nltcs", "dna", "jester", "plants", "accidents", "baudio", "bnetflix"]:
    #     dataset = get_binary_data(bname)
    #     ds_name, data, train, test, words, statistical_type, distribution_family = dataset
    #     print(ds_name, data.shape)
    #
    #     ds_context = Context()
    #     ds_context.statistical_type = statistical_type
    #     ds_context.distribution_family = distribution_family
    #     add_domains(data, ds_context)
    #
    #     #spn = learn(train, ds_context, True)
    #
    #     #print(get_structure_stats(spn))
    #     #print(np.mean(log_likelihood(spn, test, Histogram_Likelihoods)))
    #
    # 0/0
    run_experiment(get_binary_data("nltcs"), 16, True)
    run_experiment(get_binary_data("dna"), 45, True)
    run_experiment(get_binary_data("dna"), 90, True)
    run_experiment(get_binary_data("dna"), 135, True)
    run_experiment(get_binary_data("dna"), 180, True)
    run_experiment(get_binary_data("jester"), 40, True)
    run_experiment(get_binary_data("jester"), 80, True)
    run_experiment(get_binary_data("jester"), 100, True)
    run_experiment(get_binary_data("plants"), 20, True)
    run_experiment(get_binary_data("plants"), 40, True)
    run_experiment(get_binary_data("plants"), 50, True)
    run_experiment(get_binary_data("plants"), 69, True)
    run_experiment(get_binary_data("accidents"), 20, True)
    run_experiment(get_binary_data("accidents"), 40, True)
    run_experiment(get_binary_data("accidents"), 60, True)
    run_experiment(get_binary_data("accidents"), 80, True)
    run_experiment(get_binary_data("accidents"), 100, True)
    run_experiment(get_binary_data("accidents"), 111, True)
    run_experiment(get_binary_data("baudio"), 20, True)
    run_experiment(get_binary_data("baudio"), 40, True)
    run_experiment(get_binary_data("baudio"), 60, True)
    run_experiment(get_binary_data("baudio"), 80, True)
    run_experiment(get_binary_data("baudio"), 100, True)
    run_experiment(get_binary_data("bnetflix"), 20, True)
    run_experiment(get_binary_data("bnetflix"), 40, True)
    run_experiment(get_binary_data("bnetflix"), 60, True)
    run_experiment(get_binary_data("bnetflix"), 80, True)
    run_experiment(get_binary_data("bnetflix"), 100, True)

    0/0

    for topn in [5,10,20,30,40,50,60,70,80]:
        run_experiment(get_nips_data(), topn)



