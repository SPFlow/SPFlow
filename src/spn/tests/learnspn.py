'''
Created on March 20, 2018

@author: Alejandro Molina
'''
import csv

from joblib import Memory

from spn.algorithms import Inference
from spn.leaves.Histograms import add_domains, create_histogram_leaf, Histogram_Likelihoods
from spn.algorithms.StructureLearning import LearnStructure, next_operation, Context
from spn.algorithms.splitting.RDC import split_cols_RDC, split_rows_RDC




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
    print(Inference.log_likelihood(spn, nips[0:100, :], Histogram_Likelihoods))
