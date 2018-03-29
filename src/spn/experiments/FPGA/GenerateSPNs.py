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
from spn.leaves.Histograms import add_domains, create_histogram_leaf, histogram_to_str, histogram_likelihood, \
    Histogram_str_to_spn
from spn.structure.Base import Context, Product, get_nodes_by_type, Sum, Leaf

np.set_printoptions(precision=30)

memory = Memory(cachedir="cache", verbose=0, compress=9)


@memory.cache
def learn(data, ds_context, min_instances_slice=200, threshold=0.3, linear=False):
    split_cols = lambda data, ds_context, scope: split_cols_RDC(data, ds_context, scope, threshold=threshold, linear=linear)
    nextop = lambda data, no_clusters=False, no_independencies=False, is_first=False, cluster_first=True, \
                    cluster_univariate=False, min_instances_slice=min_instances_slice: next_operation(data, no_clusters,
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


def fpga_count_ops(node):
    #we want to be around 25 ads and 250 muls
    adds = 0
    muls = 0

    sum_nodes = get_nodes_by_type(node, Sum)

    for s in sum_nodes:
        adds += len(s.children)-1
        muls += len(s.weights)

    prod_nodes = get_nodes_by_type(node, Product)

    for s in prod_nodes:
        muls += len(s.children)-1

    print( "s_nodes", len(sum_nodes), "p_nodes" , len(prod_nodes), "leaf_ndoes", len(get_nodes_by_type(node, Leaf)))

    return int(adds), muls


def save_exp(spn, ds_name, size, words, data):
    print(get_structure_stats(spn))

    path = os.path.dirname(__file__)
    outprefix = path + "/spns/%s_%s/" % (ds_name, size)

    if not os.path.exists(outprefix):
        os.makedirs(outprefix)

    with open(outprefix + "eqq.txt", "w") as text_file:
        print(to_str_equation(spn, histogram_to_str, words), file=text_file)

    with open(outprefix + "spn.txt", "w") as text_file:
        print(to_str_ref_graph(spn, histogram_to_str, words), file=text_file)

    with codecs.open(outprefix + "spn.json", "w", "utf-8-sig") as text_file:
        text_file.write(to_JSON(spn))

    with codecs.open(outprefix + "stats.txt", "w", "utf-8-sig") as text_file:
        text_file.write(get_structure_stats(spn))
        text_file.write("\n")
        text_file.write("ads=%s \t muls=%s\n" % fpga_count_ops(spn))

    np.savetxt(outprefix + "all_data.txt", data, delimiter=";", header=";".join(words))



def run_experiment(dataset, top_n_features, linear=False):
    ds_name, data, train, _, words, statistical_type, _ = dataset

    data = data[:, 0:top_n_features]
    words = words[0:top_n_features]
    train = train[:, 0:top_n_features]

    ds_context = Context()
    ds_context.statistical_type = statistical_type
    add_domains(data, ds_context)


    spn = learn(train, ds_context, linear)
    save_exp(spn, ds_name, top_n_features, words, data)

def run_experiment_binary(ds_file, min_instances=200, threshold=0.3):
    ds_name, data, train, _, words, statistical_type, _ = get_binary_data(ds_file)

    ds_context = Context()
    ds_context.statistical_type = statistical_type
    add_domains(data, ds_context)

    print("train data shape" , train.shape)
    spn = learn(train, ds_context, min_instances, threshold, True)

    print(fpga_count_ops(spn))

    save_exp(spn, ds_name, min_instances, words, data)


@memory.cache
def load(eq, words):
    return str_to_spn(eq, words, Histogram_str_to_spn)

def load_spn_from_file(outprefix):
    with open(outprefix+'eqq.txt', 'r') as myfile:
        eq = myfile.read()
    with open(outprefix+'all_data.txt', 'r') as myfile:
        words = myfile.readline().strip()
        words = words[2:]
        words = words.split(';')

    spn = load(eq, words)

    return spn, words, eq





if __name__ == '__main__':

    for topn in [5,10,20,30,40,50,60,70,80]:
        run_experiment(get_nips_data(), topn)

    run_experiment_binary("msnbc", 300, 0.13) # 17, 102
    run_experiment_binary("msnbc", 200, 0.12) # 30, 165

    run_experiment_binary("bnetflix", 4000, 0.35)  # 11, 231
    run_experiment_binary("bnetflix", 1000, 0.3)  # 27, 291

    run_experiment_binary("baudio", 1000, 0.3)  # 18, 296
    run_experiment_binary("baudio", 2000, 0.3) #14, 281
    run_experiment_binary("baudio", 4000, 0.3)  # 12, 275

    run_experiment_binary("accidents", 4000, 0.42) #27, 217

    run_experiment_binary("plants", 4000, 0.5) # 14, 256

    run_experiment_binary("jester", 2000, 0.25)  # 18, 286
    run_experiment_binary("jester", 600, 0.25) #23, 302

    run_experiment_binary("dna", 800, 0.55) #2, 363

    run_experiment_binary("nltcs", 200, 0.215) #27, 152






