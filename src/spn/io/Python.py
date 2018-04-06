'''
Created on March 22, 2018

@author: Alejandro Molina
'''
import dill as pickle

def dump_python_evaluator(node, leaf_ll):
    from spn.algorithms.Inference import likelihood

    return pickle.dumps([node, leaf_ll, likelihood])

def load_from_python_dump(dump):
    node, leaf_ll, log_likelihood = pickle.loads(dump)

    return node, leaf_ll, log_likelihood