"""
Created on June 06, 2018

@author: Alejandro Molina
"""
import itertools

import time

from observations import mnist
from sklearn.preprocessing import StandardScaler

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.Validity import is_valid
from spn.experiments.RandomSPNs.region_graph import RegionGraph
from spn.experiments.RandomSPNs.RAT_SPN import RatSpn, SpnArgs
from spn.structure.Base import Sum, Product, assign_ids
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian
import numpy as np
import tensorflow as tf


def Make_SPN_from_RegionGraph(rg_layers, rgn, num_classes, num_gauss, num_sums, default_mean=0.0, default_stdev=1.0):
    def add_to_map(given_map, key, item):
        existing_items = given_map.get(key, [])
        given_map[key] = existing_items + [item]

    region_distributions = {}
    region_products = {}
    vector_list = [[]]
    for leaf_region in rg_layers[0]:
        gauss_vector = []
        for _ in range(num_gauss):
            prod = Product()
            prod.scope.extend(leaf_region)
            for r in leaf_region:
                prod.children.append(Gaussian(mean=rgn.randn(1)[0], stdev=default_stdev, scope=[r]))
                # prod.children.append(Gaussian(mean=0, stdev=default_stdev, scope=[r]))

            assert len(prod.children) > 0
            gauss_vector.append(prod)

        vector_list[-1].append(gauss_vector)
        region_distributions[leaf_region] = gauss_vector

    for layer_idx in range(1, len(rg_layers)):
        vector_list.append([])
        if layer_idx % 2 == 1:
            partitions = rg_layers[layer_idx]
            for i, partition in enumerate(partitions):
                input_regions = list(partition)
                input1 = region_distributions[input_regions[0]]
                input2 = region_distributions[input_regions[1]]

                prod_vector = []
                for c1 in input1:
                    for c2 in input2:
                        prod = Product()
                        prod.children.append(c1)
                        prod.children.append(c2)
                        prod.scope.extend(c1.scope)
                        prod.scope.extend(c2.scope)
                        prod_vector.append(prod)

                        assert len(prod.children) > 0

                vector_list[-1].append(prod_vector)

                resulting_region = frozenset(input_regions[0] | input_regions[1])
                add_to_map(region_products, resulting_region, prod_vector)
        else:
            cur_num_sums = num_classes if layer_idx == len(rg_layers) - 1 else num_sums

            regions = rg_layers[layer_idx]
            for i, region in enumerate(regions):
                product_vectors = list(itertools.chain.from_iterable(region_products[region]))

                sum_vector = []

                for _ in range(cur_num_sums):
                    sum_node = Sum()
                    sum_node.scope.extend(region)
                    sum_node.children.extend(product_vectors)
                    sum_vector.append(sum_node)
                    sum_node.weights.extend(rgn.dirichlet([1] * len(sum_node.children), 1)[0].tolist())
                    # w = np.array([1] * len(sum_node.children))
                    # w = w / np.sum(w)
                    # sum_node.weights.extend(w.tolist())

                    assert len(sum_node.children) > 0

                vector_list[-1].append(sum_vector)

                region_distributions[region] = sum_vector

    tmp_root = Sum()
    tmp_root.children.extend(vector_list[-1][0])
    tmp_root.scope.extend(tmp_root.children[0].scope)
    tmp_root.weights = [1 / len(tmp_root.children)] * len(tmp_root.children)
    assign_ids(tmp_root)

    v, err = is_valid(tmp_root)
    assert v, err
    return vector_list, tmp_root


if __name__ == "__main__":
    # rg = RegionGraph(range(3 * 3))
    rg = RegionGraph(range(28 * 28))
    for _ in range(0, 2):
        # for _ in range(0, 20):
        rg.random_split(2, 2)

    rg_layers = rg.make_layers()

    num_classes = 10

    vector_list, tmp_root = Make_SPN_from_RegionGraph(
        rg_layers, np.random.RandomState(100), num_classes=num_classes, num_gauss=5, num_sums=5
    )
    args = SpnArgs()
    args.num_gauss = 5
    args.num_sums = 5

    spns = vector_list[-1][0]
    tensor_spn = RatSpn(10, vector_list=vector_list, args=args, name="tensor-spn-from-vectorlist")
    input_ph = tf.placeholder(tf.float32, (1000, 28 * 28))
    output = tensor_spn.forward(input_ph)

    (train_im, train_lab), (test_im, test_lab) = mnist("data/mnist")
    scalar = StandardScaler().fit(train_im)
    train_im = scalar.transform(train_im)
    test_im = scalar.transform(test_im)

    add_parametric_inference_support()

    for spn in spns:
        valid, err = is_valid(spn)
        print(valid, err)
        print(get_structure_stats(spn))

        print("starting")
        tfstart = time.perf_counter()
        log_likelihood(spn, train_im[0:1000, :])
        end = time.perf_counter()
        print("finished: ", (end - tfstart))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("starting")
        tfstart = time.perf_counter()
        sess.run(output, feed_dict={input_ph: train_im[0:1000]})
        end = time.perf_counter()
        print("finished: ", (end - tfstart))

        0 / 0
