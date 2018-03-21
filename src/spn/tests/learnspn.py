'''
Created on March 20, 2018

@author: Alejandro Molina
'''
import csv

from joblib import Memory

from src.spn.algorithms.Inference import likelihood
from src.spn.leaves.Histograms import add_domains, create_histogram_leaf
from src.spn.algorithms.StructureLearning import LearnStructure, next_operation, Context
from src.spn.algorithms.splitting.RDC import split_cols_RDC, split_rows_RDC
from src.spn.io.Dumper import to_str_equation


memory = Memory(cachedir="/tmp", verbose=0, compress=9)

@memory.cache
def learn(data, ds_context):
    splitcols = lambda data, ds_context, scope: split_cols_RDC(data, ds_context, scope, threshold=0.3)

    spn = LearnStructure(data, ds_context, next_operation, split_rows_RDC, splitcols, create_histogram_leaf)

    return spn

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

    ds_context = Context()
    ds_context.statistical_type = np.asarray(["discrete"] * nips.shape[1])

    add_domains(nips, ds_context)

    spn = learn(nips, ds_context)

    #print(to_str_equation(spn, words))

    print(likelihood(spn, nips[0:100, :]))
