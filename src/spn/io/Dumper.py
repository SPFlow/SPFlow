'''
Created on March 21, 2018

@author: Alejandro Molina
'''

from src.spn.structure.Base import Product, Sum



to_str_equation_lambdas = {}

def to_str_equation(node, feature_names=None):

    if isinstance(node, Product):
        return "(" + " * ".join(map(lambda child: to_str_equation(child, feature_names), node.children)) + ")"

    if isinstance(node, Sum):
        sumeq = " + ".join(map(lambda i: str(node.weights[i]) + "*(" + to_str_equation(node.children[i], feature_names) + ")",
                                    range(len(node.children))))
        return "(" + sumeq + ")"

    tnode = type(node)
    if tnode in to_str_equation_lambdas:
        return to_str_equation_lambdas[tnode](node, feature_names)

    raise Exception('Node type not registered: ' + str(type(node)))





