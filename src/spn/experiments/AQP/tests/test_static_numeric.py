'''
Created on June 21, 2018

@author: Moritz
'''
import unittest

import numpy as np

from spn.experiments.AQP.leaves.static.StaticNumeric import create_static_leaf
from spn.experiments.AQP.leaves.static import Inference, InferenceRange, SamplingRange


from spn.experiments.AQP.Ranges import NumericRange

class TestStatic(unittest.TestCase):
    
    def test_inference(self):
        
        val = 20
        scope = [0]
        node = create_static_leaf(val, scope)
        
        
        evidence = [[20.]]
        np_evd = np.array(evidence, np.float64) 
        prob = Inference.static_likelihood(node, np_evd)[0][0]
        self.assertEqual(prob, 1)
        
        evidence = [[np.nan]]
        np_evd = np.array(evidence, np.float64) 
        prob = Inference.static_likelihood(node, np_evd)[0][0]
        self.assertEqual(prob, 1)
        
        evidence = [[29]]
        np_evd = np.array(evidence, np.float64) 
        prob = Inference.static_likelihood(node, np_evd)[0][0]
        self.assertEqual(prob, 0)
        
        evidence = [[19]]
        np_evd = np.array(evidence, np.float64) 
        prob = Inference.static_likelihood(node, np_evd)[0][0]
        self.assertEqual(prob, 0)
        
        
        evidence = [[20.00001]]
        np_evd = np.array(evidence, np.float64) 
        prob = Inference.static_likelihood(node, np_evd)[0][0]
        self.assertEqual(prob, 0)
        
        evidence = [[20.00001],[np.nan],[20],[22]]
        np_evd = np.array(evidence, np.float64) 
        prob = Inference.static_likelihood(node, np_evd)
        self.assertEqual(prob[0][0], 0)
        self.assertEqual(prob[1][0], 1)
        self.assertEqual(prob[2][0], 1)
        self.assertEqual(prob[3][0], 0)
        
        
    
    
    def test_inference_range(self):
        
        val = 20
        scope = [0]
        node = create_static_leaf(val, scope)
        
        
        rang = [NumericRange([[20]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 1)
        
        rang = [NumericRange([[19.2]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 0)
        
        rang = [NumericRange([[20.0003]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 0)
        
        rang = [NumericRange([[0]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 0)
        
        
        rang = [NumericRange([[0, 10]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 0)
        
        rang = [NumericRange([[0, 200]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 1)
        
        rang = [NumericRange([[19.99999, 20.11]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 1)
        
        rang = [NumericRange([[19.99999, 20]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 1)
        
        rang = [NumericRange([[20, 20.321]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 1)
        
        rang = [NumericRange([[19, 19.5], [20.5, 21]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 0)
        
        rang = [NumericRange([[19, 19.5], [19.999, 20.111], [20.5, 21]])]
        ranges = np.array([rang])
        prob = InferenceRange.static_likelihood_range(node, ranges)[0][0]
        self.assertAlmostEqual(prob, 1)
        
        
    
    def test_sample_range(self):
        
        val = 20
        scope = [0]
        node = create_static_leaf(val, scope)
        

        samples = SamplingRange.sample_static_node(node, 10)
        self.assertAlmostEqual(np.average(samples), 20)
        
        rang = NumericRange([[20, 20.321]])
        ranges = np.array([rang])
        samples = SamplingRange.sample_static_node(node, 10, ranges=ranges)
        self.assertAlmostEqual(np.average(samples), 20)
        
        rang = NumericRange([[19, 20]])
        ranges = np.array([rang])
        samples = SamplingRange.sample_static_node(node, 10, ranges=ranges)
        self.assertAlmostEqual(np.average(samples), 20)
        
        rang = NumericRange([[19, 19.5], [19.999, 20.111], [20.5, 21]])
        ranges = np.array([rang])
        samples = SamplingRange.sample_static_node(node, 10, ranges=ranges)
        self.assertAlmostEqual(np.average(samples), 20)
        
        rang = NumericRange([[19, 19.5]])
        ranges = np.array([rang])
        samples = SamplingRange.sample_static_node(node, 10, ranges=ranges)
        self.assertTrue(all(np.isnan(samples)))


if __name__ == '__main__':
    unittest.main()
