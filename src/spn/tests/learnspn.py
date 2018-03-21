'''
Created on March 20, 2018

@author: Alejandro Molina
'''
import csv

from src.spn.leaves.Histograms import add_domains, create_histogram_leaf
from src.spn.algorithms.StructureLearning import LearnStructure, next_operation
from src.spn.algorithms.splitting.RDC import split_cols_RDC, split_rows_RDC
from src.spn.io.Dumper import to_str_equation

if __name__ == '__main__':
    import numpy as np
    import os

    path = os.path.dirname(__file__)

    p = path + "/../../../data/nips100.csv"

    with open(p) as csvFile:
        reader = csv.reader(csvFile)
        words = next(reader)

    print(p)
    print(words)

    nips = np.loadtxt(p, skiprows=1, delimiter=',')

    print(nips)

    ds_context = type('', (object,), {})()
    ds_context.statistical_type = np.asarray(["discrete"] * nips.shape[1])

    add_domains(nips, ds_context)

    splitcols = lambda local_data, ds_context, scope: split_cols_RDC(local_data, ds_context, scope, threshold=0.3)

    spn = LearnStructure(nips, ds_context, next_operation, split_rows_RDC, splitcols, create_histogram_leaf)

    print(to_str_equation(spn, words))
