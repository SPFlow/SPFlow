"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""

from spn.algorithms.MPE import get_node_funtions
from spn.algorithms.Inference import  likelihood, max_likelihood, log_likelihood, sum_likelihood, prod_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import get_nodes_by_type, Max, Leaf, Sum, Product, get_topological_order_layers
from spn.structure.leaves.histogram.Inference import histogram_likelihood
from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
import numpy as np


def meu_sum(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    meu_children = meu_per_node[:,[child.id for child in node.children]]
    likelihood_children = lls_per_node[:,[child.id for child in node.children]]
    weighted_likelihood = np.array(node.weights)*likelihood_children
    norm = np.sum(weighted_likelihood, axis=1)
    normalized_weighted_likelihood = weighted_likelihood / norm.reshape(-1,1)
    meu_per_node[:,node.id] = np.sum(meu_children * normalized_weighted_likelihood, axis=1)

def meu_prod(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    # product node adds together the utilities of its children
    # if there is only one utility node then only one child of each product node
    # will have a utility value
    meu_children = meu_per_node[:,[child.id for child in node.children]]
    meu_per_node[:,node.id] = np.nansum(meu_children,axis=1)

def meu_max(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    meu_children = meu_per_node[:, [child.id for child in node.children]]
    decision_value_given = data[:, node.dec_idx]
    max_value = node.dec_values[np.argmax(meu_children, axis=1)]
    # if data contains a decision value use that otherwise use max
    dec_value = np.select([np.isnan(decision_value_given), True],
                          [max_value, decision_value_given]).astype(int)
    dec_value_to_child_id = lambda val: node.children[list(node.dec_values).index(val)].id
    dec_value_to_child_id = np.vectorize(dec_value_to_child_id)
    child_id = dec_value_to_child_id(dec_value)
    meu_per_node[:,node.id] = meu_per_node[np.arange(meu_per_node.shape[0]),child_id]

def meu_util(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    #returns average value of the utility node
    util_value = 0
    for i in range(len(node.bin_repr_points)):
        util_value += node.bin_repr_points[i] * node.densities[i]
    utils = np.empty((data.shape[0]))
    utils[:] = util_value
    meu_per_node[:,node.id] = utils * lls_per_node[:,node.id]


_node_bottom_up_meu = {Sum: meu_sum, Product: meu_prod, Max: meu_max, Utility: meu_util}

def meu(root, input_data,
        node_bottom_up_meu=_node_bottom_up_meu,
        in_place=False):
    # valid, err = is_valid(node)
    # assert valid, err
    if in_place:
        data = input_data
    else:
        data = np.copy(input_data)
    nodes = get_nodes_by_type(root)
    utility_scope = set()
    for node in nodes:
        if type(node) is Utility:
            utility_scope.add(node.scope[0])
    assert np.all(np.isnan(data[:, list(utility_scope)])), "Please specify all utility values as np.nan"
    likelihood_per_node = np.zeros((data.shape[0], len(nodes)))
    meu_per_node = np.zeros((data.shape[0], len(nodes)))
    meu_per_node.fill(np.nan)
    # one pass bottom up evaluating the likelihoods
    likelihood(root, data, dtype=data.dtype, lls_matrix=likelihood_per_node)
    eval_spmn_bottom_up_meu(
            root,
            _node_bottom_up_meu,
            meu_per_node=meu_per_node,
            data=data,
            lls_per_node=likelihood_per_node
        )
    result = meu_per_node[:,root.id]
    return result


def eval_spmn_bottom_up_meu(root, eval_functions, meu_per_node=None, data=None, lls_per_node=None):
    """
      evaluates an spn top to down
      :param root: spnt root
      :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, [parent_results], args**) and returns {child : intermediate_result}. This intermediate_result will be passed to child as parent_result. If intermediate_result is None, no further propagation occurs
      :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
      :param parent_result: initial input to the root node
      :param args: free parameters that will be fed to the lambda functions.
      :return: the result of computing and propagating all the values throught the network, decisions at each max node for the instances reaching that max node.
      """
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)
    for layer in get_topological_order_layers(root):
        for n in layer:
            if type(n)==Max or type(n)==Sum or type(n)==Product or type(n)==Utility:
                func = n.__class__._eval_func[-1]
                func(n, meu_per_node, data=data, lls_per_node=lls_per_node)
    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")


def eval_spmn_top_down_meu(root, eval_functions,
        all_results=None, parent_result=None, data=None,
        lls_per_node=None, likelihood_per_node=None):
    """
      evaluates an spn top to down


      :param root: spnt root
      :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, [parent_results], args**) and returns {child : intermediate_result}. This intermediate_result will be passed to child as parent_result. If intermediate_result is None, no further propagation occurs
      :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
      :param parent_result: initial input to the root node
      :param args: free parameters that will be fed to the lambda functions.
      :return: the result of computing and propagating all the values throught the network, decisions at each max node for the instances reaching that max node.
      """
    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    all_decisions = []
    all_max_nodes = []
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)

    all_results[root] = [parent_result]

    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]

            param = all_results[n]
            if type(n) == Max:
                result, decision_values, max_nodes = func(n, param,
                                    data=data, lls_per_node=lls_per_node)
                all_decisions.append(decision_values)
                all_max_nodes.append(max_nodes)
            elif type(n) == Sum:
                result = func(n, param,
                        likelihood_per_node=likelihood_per_node,
                        data=data,
                        lls_per_node=lls_per_node
                    )
            else:
                result = func(n, param, data=data, lls_per_node=lls_per_node)

            if result is not None and not isinstance(n, Leaf):
                assert isinstance(result, dict)

                for child, param in result.items():
                    if child not in all_results:
                        all_results[child] = []
                    all_results[child].append(param)

    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    return all_results[root], all_decisions, all_max_nodes

def best_next_decision(root, input_data, in_place=False):
    if in_place:
        data = input_data
    else:
        data = np.copy(input_data)
    nodes = get_nodes_by_type(root)
    dec_dict = {}
    # find all possible decision values
    for node in nodes:
        if type(node) == Max:
            if node.dec_idx in dec_dict:
                dec_dict[node.dec_idx].union(set(node.dec_values))
            else:
                dec_dict[node.dec_idx] = set(node.dec_values)
    next_dec_idx = None
    # find next undefined decision
    for idx in dec_dict.keys():
        if np.all(np.isnan(data[:,idx])):
            next_dec_idx = idx
            break
    assert next_dec_idx != None, "please assign all values of next decision to np.nan"
    # determine best decisions based on meu
    dec_vals = list(dec_dict[next_dec_idx])
    best_decisions = np.full((1,data.shape[0]),dec_vals[0])
    data[:,next_dec_idx] = best_decisions
    meu_best = meu(root, data)
    for i in range(1, len(dec_vals)):
        decisions_i = np.full((1,data.shape[0]), dec_vals[i])
        data[:,next_dec_idx] = decisions_i
        meu_i = meu(root, data)
        best_decisions = np.select([np.greater(meu_i, meu_best),True],[decisions_i, best_decisions])
        meu_best = np.maximum(meu_i,meu_best)
    return best_decisions
