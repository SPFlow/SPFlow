import logging
import sys


import numpy as np

from spn.algorithms.LearningWrappers import learn_rand_spn
from spn.structure.StatisticalTypes import MetaType, Type
from spn.structure.Base import Context
from spn.structure.Base import Leaf, get_nodes_by_type

from spn.structure.leaves.parametric.Parametric import Gaussian, Gamma, LogNormal, Categorical, Poisson, Parametric, Bernoulli, Geometric, Hypergeometric, NegativeBinomial, Exponential, type_mixture_leaf_factory, LEAF_TYPES
from spn.structure.leaves.parametric.Text import add_parametric_text_support
from spn.io.Text import to_JSON, spn_to_str_equation

from spn.algorithms.Sampling import sample_instances
from visualize import visualize_data_partition, reorder_data_partitions
from spn.algorithms.Statistics import get_structure_stats_dict


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    N = 1000
    D = 4

    rand_gen = np.random.RandomState(17)
    data = rand_gen.normal(loc=0, scale=1, size=(N, D))

    meta_types = np.array([MetaType.REAL, MetaType.DISCRETE, MetaType.REAL, MetaType.DISCRETE])
    ds_context = Context(meta_types=meta_types)

    types = np.array([Type.REAL, Type.COUNT, Type.POSITIVE, Type.COUNT])
    ds_context.types = types

    type_param_map = {Type.REAL: [(Gaussian, {'mean': 5, 'stdev': 5}),
                                  (Gaussian, {'mean': 100, 'stdev': 2}),
                                  (Gaussian, {'mean': 30, 'stdev': 1.3})],
                      Type.COUNT: [(Geometric, {'p': 0.02}),
                                   (Geometric, {'p': 0.22}),
                                   (Poisson, {'mean': 10})],
                      Type.POSITIVE: [(Gamma, {'alpha': 20, 'beta': 5}),
                                      (Gamma, {'alpha': 20, 'beta': 2}),
                                      (Exponential, {'l': 5})],
                      }

    ds_context.param_form_map = type_param_map

    spn = learn_rand_spn(data,
                         ds_context,
                         min_instances_slice=500,
                         row_a=2, row_b=5,
                         col_a=2, col_b=5,
                         col_threshold=0.3,
                         memory=None, rand_gen=rand_gen)

    add_parametric_text_support()
    print(spn_to_str_equation(spn))
    print(spn.scope)

    #
    # sampling again
    X, _Z, P = sample_instances(spn, D, N, rand_gen, return_Zs=True,
                                return_partition=True, dtype=np.float64)

    #
    # visualizing
    stats = get_structure_stats_dict(spn)
    inv_leaf_map = {l.id: spn_to_str_equation(l)  # l.__class__.__name__
                    for l in get_nodes_by_type(spn, Leaf)}
    title_str = "{} samples from spn with {} sums {} prods {} leaves".format(N,
                                                                             stats['sum'],
                                                                             stats['prod'],
                                                                             stats['leaf'])
    visualize_data_partition(P, color_map_ids=inv_leaf_map, title=title_str)
    #
    # ordering partitions
    reord_ids = reorder_data_partitions(P)
    title_str = "ordered {} samples from spn with {} sums {} prods {} leaves".format(N,
                                                                                     stats['sum'],
                                                                                     stats['prod'],
                                                                                     stats['leaf'])
    visualize_data_partition(P[reord_ids], color_map_ids=inv_leaf_map, title=title_str)
