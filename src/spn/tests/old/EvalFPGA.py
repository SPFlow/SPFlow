'''
Created on March 22, 2018

@author: Alejandro Molina
'''
import numpy as np
from sklearn.cross_validation import train_test_split

from spn.algorithms import Inference
from spn.algorithms.Statistics import get_structure_stats
from spn.data.datasets import get_nips_data
from spn.io.Text import str_to_spn

if __name__ == '__main__':
    with open('40_eqq.txt', 'r') as myfile:
        eq = myfile.read()
    with open('40_testdata.txt', 'r') as myfile:
        words = myfile.readline().strip()
        words = words[2:]
        words = words.split(';')

    # print(eq)
    print(words)

    spn = str_to_spn(eq, words)

    print(get_structure_stats(spn))

    # print(Text.toJSON(spn))

    data = np.loadtxt("40_testdata.txt", delimiter=';')

    ll = Inference.likelihood(spn, data)

    print(ll)
    print("average LL", np.mean(ll))

    ds_name, words, data, _, _, _, _ = get_nips_data()

    top_n_features = 40

    train, test = train_test_split(data[:, 0:top_n_features], test_size=0.2, random_state=42)

    ll = Inference.likelihood(spn, test)
    print("average LL2", np.mean(ll))
