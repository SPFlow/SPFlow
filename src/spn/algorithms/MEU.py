"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""

from spn.algorithms.MPE import get_node_funtions
from spn.algorithms.Inference import  likelihood, max_likelihood, log_likelihood, sum_likelihood, prod_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import get_nodes_by_type, Max, Leaf, Sum, Product, get_topological_order_layers

import numpy as np
from collections import defaultdict
import collections


def merge_input_vals(l):
    return np.concatenate(l)

def meu_max(node, parent_result, data=None, lls_per_node=None, rand_gen=None):
    if len(parent_result) == 0:
        return None

    parent_result = merge_input_vals(parent_result)

    w_children_log_probs = np.zeros((len(parent_result), len(node.dec_values)))
    for i, c in enumerate(node.children):
        w_children_log_probs[:, i] = lls_per_node[parent_result, c.id]

    max_child_branches = np.argmax(w_children_log_probs, axis=1)

    children_row_ids = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = parent_result[max_child_branches == i]
    # print("node and max_child_branches", (node.var_name, max_child_branches))
    ## decision values at each node

    decision_values = {}
    decision_values[node.feature_name] = np.column_stack((parent_result, max_child_branches))

    # print("w_children_log_probs",w_children_log_probs)
    return children_row_ids, decision_values



node_functions =  get_node_funtions()
_node_top_down_meu= node_functions[0].copy()
_node_bottom_up_meu = node_functions[1].copy()
_node_top_down_meu.update({Max:meu_max})
_node_bottom_up_meu.update({Sum: sum_likelihood, Product: prod_likelihood, Max:max_likelihood})




def meu(node, input_data, node_top_down_meu=_node_top_down_meu, node_bottom_up_meu=_node_bottom_up_meu, in_place=False):
    valid, err = is_valid(node)
    assert valid, err
    if in_place:
        data = input_data
    else:
        data = np.array(input_data)

    nodes = get_nodes_by_type(node)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    # one pass bottom up evaluating the likelihoods
    # log_likelihood(node, data, dtype=data.dtype, node_log_likelihood=node_bottom_up_meu, lls_matrix=lls_per_node)
    likelihood(node, data, dtype=data.dtype, node_likelihood=node_bottom_up_meu, lls_matrix=lls_per_node)

    meu_val = lls_per_node[:, 0]

    instance_ids = np.arange(data.shape[0])

    # one pass top down to decide on the max branch until it reaches a leaf; returns  all_result, decisions at each max node for each instance.
    all_result, all_decisions = eval_spn_top_down_meu(node, node_top_down_meu, parent_result=instance_ids, data=data,
                                                      lls_per_node=lls_per_node)

    decisions = merge_rows_for_decisions(all_decisions)

    return meu_val, decisions



def merge_rows_for_decisions(all_decisions=None):
    """
    merges different values of same key into one key:value pair
    :param all_decisions: takes the dictionary of decisions with key as max_node name and values as numpy array with first column as row number of instance, second column as decision.
    :return: merged values of same key and sorted on rowNum, key:value pairs(max_naode:[[rowNum, decision]]
    """

    decisions = defaultdict(list)

    for dict_decisions in all_decisions:
        for decision_node, decision in dict_decisions.items():
            decisions[decision_node].append(decision)

    # print(decisions)

    for decision_node, decision in decisions.items():
        decisions[decision_node] = np.concatenate(tuple(decision))

    # print(decisions)

    for decision_node, decision in decisions.items():
        decisions[decision_node] = decision[decision[:, 0].argsort()]

    return decisions


def eval_spn_top_down_meu(root, eval_functions, all_results=None, parent_result=None, **args):
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
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)

    all_results[root] = [parent_result]

    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]

            param = all_results[n]
            if type(n) != Max:
                result = func(n, param, **args)
            else:
                result, decision_values = func(n, param, **args)
                all_decisions.append(decision_values)


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

    return all_results[root], all_decisions