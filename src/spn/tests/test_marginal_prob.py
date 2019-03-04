import unittest

from spn.algorithms.MarginalProb import get_marginal_prob_bernoulli
from spn.structure.leaves.parametric.Parametric import Bernoulli
from spn.algorithms.Sampling import sample_instances
from numpy.random.mtrand import RandomState

import numpy as np

class TestMarginalProb(unittest.TestCase):

    def test_marginal_prob_bernoulli(self): 
        D = Bernoulli(p=0.2, scope=[0])
        E = Bernoulli(p=0.6, scope=[1])
        F = Bernoulli(p=0.4, scope=[0])
        G = Bernoulli(p=0.7, scope=[1])

        B = D * E
        C = F * G

        A = 0.3 * B + 0.7 * C

        num_samples = 1000

        input_placeholder_random = np.array( [np.nan] * ( len( A.scope ) * num_samples ) ).reshape(-1, len( A.scope )) 

        # This is the heuristic probability of drawing true at an input. 
        random_samples = sample_instances(A, input_placeholder_random, RandomState(123))
        mean_output = np.mean(random_samples, axis=0)

        # This is the anlytical probability of drawing true at an input. 
        marginal_prob = get_marginal_prob_bernoulli(A, np.array( [np.nan] * len(A.scope) ).reshape(-1, len(A.scope)) )

        # Should be pretty close. 
        self.assertTrue(np.allclose(mean_output, marginal_prob, atol=1.e-2))

if __name__ == "__main__": 

    unittest.main()

        