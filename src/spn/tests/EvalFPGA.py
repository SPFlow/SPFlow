'''
Created on March 22, 2018

@author: Alejandro Molina
'''
from spn.algorithms import Inference
from spn.algorithms.Statistics import get_structure_stats
from spn.io import Text
from spn.io.Text import str_to_spn
import numpy as np

from spn.leaves.Histograms import str_to_spn_lambdas, Histogram_Likelihoods

if __name__ == '__main__':
    with open('40_eqq.txt', 'r') as myfile:
        eq = myfile.read()
    with open('40_testdata.txt', 'r') as myfile:
        words = myfile.readline().strip()
        words = words[2:]
        words = words.split(';')

    #print(eq)
    print(words)

    spn = str_to_spn(eq, words, str_to_spn_lambdas)

    print(get_structure_stats(spn))

    #print(Text.toJSON(spn))

    data = np.loadtxt("40_testdata.txt", delimiter=';')

    ll = Inference.log_likelihood(spn, data, Histogram_Likelihoods)

    print(ll)
    print("average LL", np.mean(ll))
