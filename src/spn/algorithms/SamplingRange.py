'''
Created on May 23, 2018

@author: Moritz
'''

import numpy as np
from scipy.special import logsumexp


from spn.structure.Base import Product, Sum, Leaf, get_nodes_by_type
from spn.algorithms.Inference import _node_log_likelihood



LOG_ZERO = -300



def validate_ids(node):
    all_nodes = get_nodes_by_type(node)

    ids = set()
    for n in all_nodes:
        ids.add(n.id)

    assert len(ids) == len(all_nodes), "not all nodes have ID's"

    assert min(ids) == 0 and max(ids) == len(ids) - 1, "ID's are not in order"


def reset_node_counters(node):
    all_nodes = get_nodes_by_type(node)
    max_id = 0
    for n in all_nodes:
        # reset sum node counts
        if isinstance(n, Sum):
            n.edge_counts = np.zeros(len(n.children), dtype=np.int64)
        # sets nr_nodes to the max id
        max_id = max(max_id, n.id)
        n.row_ids = []
    return max_id


def set_weights_for_evidence(node, ranges, dtype=np.float64, context=None, node_log_likelihood=_node_log_likelihood):
    
    if isinstance(node, Product):
        '''
        Just multiply the probabilities and return the result
        '''
        prob = 0.
        for c in node.children:
            
            tmp = set_weights_for_evidence(c, ranges, context=context, node_log_likelihood=node_log_likelihood)
            
            if tmp is None:
                return None
            
            prob += tmp
        
        return prob

    elif isinstance(node, Sum):
        evidence_log_weights = np.zeros(len(node.children))
        evidence_weights = np.zeros(len(node.children))
        valid_child = False
        for i, c in enumerate(node.children):
            prob = set_weights_for_evidence(c, ranges, context=context, node_log_likelihood=node_log_likelihood)
            if prob is None:
                evidence_weights[i] = 0
            else:
                valid_child = True
                evidence_weights[i] = np.exp(prob) * node.weights[i]
        
        if not valid_child:
            return None
        
        node.evidence_weights = evidence_weights / np.sum(evidence_weights)
        
        return logsumexp(evidence_log_weights)
    
    elif isinstance(node, Leaf):
        t_node = type(node)
        if t_node in node_log_likelihood:
            ranges = np.array([ranges])
            log_prob = node_log_likelihood[t_node](node, ranges, dtype=dtype, node_log_likelihood=node_log_likelihood)
            
            if log_prob <= LOG_ZERO:
                return None
            else:
                return log_prob
            
        else:
            raise Exception('No log-likelihood method specified for node type: ' + str(type(node)))


def sample_instances(node, D, n_samples, rand_gen, ranges=None, dtype=np.float64, node_sample=None, node_log_likelihood=_node_log_likelihood):

    instance_ids = np.arange(n_samples)
    X = np.zeros((n_samples, D), dtype=dtype)

    _max_id = reset_node_counters(node)
    result = set_weights_for_evidence(node, ranges, node_log_likelihood=node_log_likelihood)
    
    if result is None:
        return np.zeros((0, D), dtype=dtype)
    
    def _sample_instances(node, row_ids):
        if len(row_ids) == 0:
            return
        node.row_ids = row_ids

        if isinstance(node, Product):
            for c in node.children:
                _sample_instances(c, row_ids)
            return

        if isinstance(node, Sum):

            rand_child_branches = np.random.choice(np.arange(len(node.evidence_weight)), p=node.evidence_weight, size=len(row_ids))
            
            for i, c in enumerate(node.children):
                new_row_ids = row_ids[rand_child_branches == i]
                node.edge_counts[i] = len(new_row_ids)
                _sample_instances(c, new_row_ids)

        if isinstance(node, Leaf):
            
            t_node = type(node)
            if t_node in node_sample:
                X[row_ids, node.scope] = node_sample[t_node](node, len(row_ids), rand_gen, ranges)
            else:
                raise Exception('No sample method specified for node type: ' + str(type(node)))

            return

    _sample_instances(node, instance_ids)
    
    return X






if __name__ == '__main__':
    
    import os 
    os.environ["R_HOME"] = "C:/Program Files/R/R-3.3.3" 
    os.environ["R_USER"] = "Moritz" 
    
    from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
    from spn.structure.leaves.piecewise.SamplingRange import sample_piecewise_node
    from spn.structure.leaves.piecewise.InferenceRange import piecewise_log_likelihood_range
    
    from spn.structure.leaves.parametric.Parametric import Categorical
    from spn.structure.leaves.parametric.SamplingRange import sample_categorical_node
    from spn.structure.leaves.parametric.InferenceRange import categorical_log_likelihood_range
    
    from spn.structure.Base import Context
    from spn.structure.StatisticalTypes import MetaType
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange
    
    from spn.algorithms import SamplingRange
    
    
    
    rand_gen = np.random.RandomState(100)
    
    #Create SPN
    node1 = Categorical(p=[0.9, 0.1], scope=[0])
    node2 = Categorical(p=[0.1, 0.9], scope=[0])
    
    
    x = [0.,  1.,  2.,  3., 4.]
    y = [0., 10., 0., 0., 0.]
    x, y = np.array(x), np.array(y)
    auc = np.trapz(y, x)
    y = y / auc
    node3 = PiecewiseLinear(x_range=x, y_range=y, bin_repr_points=x[1:-1], scope=[1])
    
    x = [0.,  1.,  2.,  3., 4.]
    y = [0., 0., 0., 10., 0.]
    x, y = np.array(x), np.array(y)
    auc = np.trapz(y, x)
    y = y / auc
    node4 = PiecewiseLinear(x_range=x, y_range=y, bin_repr_points=x[1:-1], scope=[1])
    
    root_node = 0.49 * (node1 * node3) + 0.51 * (node2 * node4)
    
       
    #Set context
    meta_types = [MetaType.DISCRETE, MetaType.REAL]
    domains = [[0,1],[0.,4.]]
    ds_context = Context(meta_types=meta_types, domains=domains)
    
    
    inference_support_ranges = {PiecewiseLinear : piecewise_log_likelihood_range, 
                                Categorical     : categorical_log_likelihood_range}
    
    node_sample = {Categorical : sample_categorical_node,
                   PiecewiseLinear : sample_piecewise_node}
    
    ranges = [NominalRange([0]),None]
    samples = SamplingRange.sample_instances(root_node, 2, 30, rand_gen, ranges=ranges, context=ds_context, node_sample=node_sample, node_log_likelihood=inference_support_ranges)#, return_Zs, return_partition, dtype)
    print("Samples: " + str(samples))
    
    ranges = [NominalRange([0]),NumericRange([[3., 3.1], [3.5, 4.]])]
    samples = SamplingRange.sample_instances(root_node, 2, 30, rand_gen, ranges=ranges, context=ds_context, node_sample=node_sample, node_log_likelihood=inference_support_ranges)#, return_Zs, return_partition, dtype)
    print("Samples: " + str(samples))
