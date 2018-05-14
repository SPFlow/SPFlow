from collections import OrderedDict

import numpy as np

from spn.algorithms.Inference import compute_global_type_weights
from spn.structure.Base import Leaf, get_nodes_by_type, Sum, Product, assign_ids, rebuild_scopes_bottom_up, compute_leaf_global_mix_weights
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.StatisticalTypes import MetaType, Type
from spn.structure.leaves.parametric.Text import add_parametric_text_support
from spn.io.Text import to_JSON, spn_to_str_equation
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support

#
# create an SPN over three random variables X_1, X_2, X_3
from spn.structure.leaves.typedleaves.Text import add_typed_leaves_text_support
from spn.structure.leaves.typedleaves.TypedLeaves import type_mixture_leaf_factory

add_typed_leaves_text_support()
add_parametric_inference_support()
#
# root is a sum
root = Sum()


#
# two product nodes
l_prod = Product()
r_prod = Product()
root.children = [l_prod, r_prod]
root.weights = np.array([0.75, 0.25])

#
# priors, but useless
pm_continuous_param_map = OrderedDict({
    Type.REAL: OrderedDict({Gaussian: {'params': {'mean': 5, 'stdev': 5},
                                       'prior': None}}),
    Type.POSITIVE: OrderedDict({Gamma: {'params': {'alpha': 20, 'beta': 5},
                                        'prior': None}, }),
})

pm_discrete_param_map = OrderedDict({
    Type.CATEGORICAL: OrderedDict({Categorical: {'params': {'p': np.array([0.1, 0.1, 0.1, 0.1, 0.6])},
                                                 'prior': None}}),
    Type.COUNT: OrderedDict({Poisson: {'params': {'mean': 10},
                                       'prior': None}, }),
})

#
# right branch, three leaves
lf_1_init_weights = {Gaussian: 0.5, Gamma: 0.5}
# lf_1_init_weights = np.array([0.5, 0.5])
fat_right_leaf_1, _priors = type_mixture_leaf_factory(leaf_type='pm',
                                                      leaf_meta_type=MetaType.REAL,
                                                      type_to_param_map=pm_continuous_param_map,
                                                      scope=[0],
                                                      init_weights=lf_1_init_weights)
lf_2_init_weights = {Gaussian: 0.2, Gamma: 0.8}
#lf_2_init_weights = np.array([0.2, 0.8])
fat_right_leaf_2, _priors = type_mixture_leaf_factory(leaf_type='pm',
                                                      leaf_meta_type=MetaType.REAL,
                                                      type_to_param_map=pm_continuous_param_map,
                                                      scope=[1],
                                                      init_weights=lf_2_init_weights)
lf_3_init_weights = {Poisson: 0.9, Categorical: 0.1}
# lf_3_init_weights = np.array([.9, .1])
fat_right_leaf_3, _priors = type_mixture_leaf_factory(leaf_type='pm',
                                                      leaf_meta_type=MetaType.DISCRETE,
                                                      type_to_param_map=pm_discrete_param_map,
                                                      scope=[2],
                                                      init_weights=lf_3_init_weights)

r_prod.children = [fat_right_leaf_1, fat_right_leaf_2, fat_right_leaf_3]

#
# left branch one leaf and one sum nodes
rf_3_init_weights = {Categorical: 0.4, Poisson: 0.6}
# rf_3_init_weights = np.array([.6, .4])
fat_left_leaf_3, _priors = type_mixture_leaf_factory(leaf_type='pm',
                                                     leaf_meta_type=MetaType.DISCRETE,
                                                     type_to_param_map=pm_discrete_param_map,
                                                     scope=[2],
                                                     init_weights=rf_3_init_weights)

left_sum_node = Sum()
l_prod.children = [left_sum_node, fat_left_leaf_3]


l_l_prod = Product()
l_r_prod = Product()
left_sum_node.children = [l_l_prod, l_r_prod]
left_sum_node.weights = np.array([0.3, 0.7])

#
# far left branch, two leaves
a_lf_1_init_weights = {Gaussian: 0.1, Gamma: 0.9}
# a_lf_1_init_weights = np.array([.1, .9])
a_fat_right_leaf_1, _priors = type_mixture_leaf_factory(leaf_type='pm',
                                                        leaf_meta_type=MetaType.REAL,
                                                        type_to_param_map=pm_continuous_param_map,
                                                        scope=[0],
                                                        init_weights=a_lf_1_init_weights)
a_lf_2_init_weights = {Gaussian: 0.5, Gamma: 0.5}
# a_lf_2_init_weights = np.array([.5, .5])
a_fat_right_leaf_2, _priors = type_mixture_leaf_factory(leaf_type='pm',
                                                        leaf_meta_type=MetaType.REAL,
                                                        type_to_param_map=pm_continuous_param_map,
                                                        scope=[1],
                                                        init_weights=a_lf_2_init_weights)
l_l_prod.children = [a_fat_right_leaf_1, a_fat_right_leaf_2]

#
# far left branch, two leaves
b_lf_1_init_weights = {Gaussian: 0.6, Gamma: 0.4}
# b_lf_1_init_weights = np.array([.6, .4])
b_fat_right_leaf_1, _priors = type_mixture_leaf_factory(leaf_type='pm',
                                                        leaf_meta_type=MetaType.REAL,
                                                        type_to_param_map=pm_continuous_param_map,
                                                        scope=[0],
                                                        init_weights=b_lf_1_init_weights)
b_lf_2_init_weights = {Gaussian: 0.3, Gamma: 0.7}
# b_lf_2_init_weights = np.array([.3, .7])
b_fat_right_leaf_2, _priors = type_mixture_leaf_factory(leaf_type='pm',
                                                        leaf_meta_type=MetaType.REAL,
                                                        type_to_param_map=pm_continuous_param_map,
                                                        scope=[1],
                                                        init_weights=b_lf_2_init_weights)
l_r_prod.children = [b_fat_right_leaf_1, b_fat_right_leaf_2]


#
# composing
rebuild_scopes_bottom_up(root)
assign_ids(root)
print(root)
print(spn_to_str_equation(root))

global_W = compute_global_type_weights(root)
print('GLOBAL_W', global_W)

global_W = compute_global_type_weights(root, aggr_type=True)
print('GLOBAL_W', global_W)

gw_map = compute_leaf_global_mix_weights(root)
print('G MIX W', gw_map)
