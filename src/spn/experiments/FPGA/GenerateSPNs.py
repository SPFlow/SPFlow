'''
Created on March 24, 2018

@author: Alejandro Molina
'''
import codecs
import os

import numpy as np
from joblib import Memory

from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.Statistics import get_structure_stats
from spn.data.datasets import get_binary_data, get_nips_data
from spn.io.Text import to_JSON, spn_to_str_equation, str_to_spn, spn_to_str_ref_graph
from spn.structure.Base import Context, Product, get_nodes_by_type, Sum, Leaf
from spn.structure.leaves.Histograms import add_domains

np.set_printoptions(precision=30)

memory = Memory(cachedir="cache", verbose=0, compress=9)


def fpga_count_ops(node):
    # we want to be around 25 ads and 250 muls
    adds = 0
    muls = 0

    sum_nodes = get_nodes_by_type(node, Sum)

    for s in sum_nodes:
        adds += len(s.children) - 1
        muls += len(s.weights)

    prod_nodes = get_nodes_by_type(node, Product)

    for s in prod_nodes:
        muls += len(s.children) - 1

    print("s_nodes", len(sum_nodes), "p_nodes", len(prod_nodes), "leaf_ndoes", len(get_nodes_by_type(node, Leaf)))

    return int(adds), muls


def save_exp(spn, ds_name, size, words, data):
    print(get_structure_stats(spn))

    path = os.path.dirname(__file__)
    outprefix = path + "/spns/%s_%s/" % (ds_name, size)

    if not os.path.exists(outprefix):
        os.makedirs(outprefix)

    with open(outprefix + "eqq.txt", "w") as text_file:
        print(spn_to_str_equation(spn, words), file=text_file)

    with open(outprefix + "spn.txt", "w") as text_file:
        print(spn_to_str_ref_graph(spn, words), file=text_file)

    with codecs.open(outprefix + "spn.json", "w", "utf-8-sig") as text_file:
        text_file.write(to_JSON(spn))

    with codecs.open(outprefix + "stats.txt", "w", "utf-8-sig") as text_file:
        text_file.write(get_structure_stats(spn))
        text_file.write("\n")
        text_file.write("ads=%s \t muls=%s\n" % fpga_count_ops(spn))

    np.savetxt(outprefix + "all_data.txt", data, delimiter=";", header=";".join(words))


def run_experiment(dataset, top_n_features, linear=False):
    ds_name, words, data, train, _, statistical_type, _ = dataset

    data = data[:, 0:top_n_features]
    words = words[0:top_n_features]
    train = train[:, 0:top_n_features]

    ds_context = Context()
    ds_context.statistical_type = statistical_type
    add_domains(data, ds_context)

    spn = learn_mspn(train, ds_context, linear=linear, memory=memory)
    save_exp(spn, ds_name, top_n_features, words, data)


def run_experiment_binary(ds_file, min_instances=200, threshold=0.3):
    ds_name, words, data, train, _, statistical_type, _ = get_binary_data(ds_file)

    ds_context = Context()
    ds_context.statistical_type = statistical_type
    add_domains(data, ds_context)

    print("train data shape", train.shape)
    spn = learn_mspn(train, ds_context, min_instances_slice=min_instances, threshold=threshold, linear=True,
                     memory=memory)

    print(fpga_count_ops(spn))

    save_exp(spn, ds_name, min_instances, words, data)


@memory.cache
def load(eq, words):
    return str_to_spn(eq, words)


def load_spn_from_file(outprefix):
    with open(outprefix + 'eqq.txt', 'r') as myfile:
        eq = myfile.read()
    with open(outprefix + 'all_data.txt', 'r') as myfile:
        words = myfile.readline().strip()
        words = words[2:]
        words = words.split(';')

    spn = load(eq, words)

    return spn, words, eq


if __name__ == '__main__':

    for topn in [5, 10, 20, 30, 40, 50, 60, 70, 80]:
        run_experiment(get_nips_data(), topn)

    run_experiment_binary("msnbc", 300, 0.13)  # 17, 102
    run_experiment_binary("msnbc", 200, 0.12)  # 30, 165

    run_experiment_binary("bnetflix", 4000, 0.35)  # 11, 231
    run_experiment_binary("bnetflix", 1000, 0.3)  # 27, 291

    run_experiment_binary("baudio", 1000, 0.3)  # 18, 296
    run_experiment_binary("baudio", 2000, 0.3)  # 14, 281
    run_experiment_binary("baudio", 4000, 0.3)  # 12, 275

    run_experiment_binary("accidents", 4000, 0.42)  # 27, 217

    run_experiment_binary("plants", 4000, 0.5)  # 14, 256

    run_experiment_binary("jester", 2000, 0.25)  # 18, 286
    run_experiment_binary("jester", 600, 0.25)  # 23, 302

    run_experiment_binary("dna", 800, 0.55)  # 2, 363

    run_experiment_binary("nltcs", 200, 0.215)  # 27, 152
