'''
Created on March 20, 2018

@author: Alejandro Molina
'''

import os

path = os.path.dirname(__file__)

if __name__ == '__main__':
    import numpy as np

    p = path + "/../../../data/nips100.csv"

    print(p)

    nips = np.loadtxt(p, skiprows=1, delimiter=',')

    print(nips)

    ds_context = type('', (object,), {})()
    ds_context.statistical_type = np.asarray(["discrete"] * nips.shape[1])

    scope = [1, 2, 3, 5, 6, 7, 8, 9, 10]

    spn = LearnStructure(nips, ds_context, next_operation, split_rows_RDC, split_cols_RDC, )
