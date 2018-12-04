"""
Created on March 27, 2018

@author: Alejandro Molina
"""

import numpy as np

if __name__ == "__main__":
    import dill as pickle

    eval_conditional = pickle.load(open("conditional.bin", "r+b"))

    ll = eval_conditional(np.array([4, 1, 5]).reshape(1, -1))

    print(ll)
