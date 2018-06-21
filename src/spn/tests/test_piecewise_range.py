import unittest
import numpy as np

from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.leaves.piecewise import InferenceRange, SamplingRange
from spn.experiments.AQP.Ranges import NumericRange

from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType


class TestPiecewiseRange(unittest.TestCase):
    
    
    def test_inference_range(self):
        
        np.random.seed(10)
        data = np.random.normal(20, scale=5, size=1000).reshape((1000,1))
        numpy_data = np.array(data, np.float64)
        meta_types = [MetaType.REAL]
        domains = [[np.min(numpy_data[:, 0]), np.max(numpy_data[:, 0])]]
        ds_context = Context(meta_types=meta_types, domains=domains)
        pwl = create_piecewise_leaf(data, ds_context, scope=[0], prior_weight=None)
        
        rang = [NumericRange([[20]])]
        ranges = np.array([rang])
        prob = InferenceRange.piecewise_likelihood_range(pwl, ranges)[0][0]
        self.assertAlmostEqual(prob, 0.086475210674)

        rang = [NumericRange([[21]])]
        ranges = np.array([rang])
        prob = InferenceRange.piecewise_likelihood_range(pwl, ranges)[0][0]
        self.assertAlmostEqual(prob, 0.0855907611968)

        rang = [NumericRange([[19]])]
        ranges = np.array([rang])
        prob = InferenceRange.piecewise_likelihood_range(pwl, ranges)[0][0]
        self.assertAlmostEqual(prob, 0.0833451329643)

        rang = [NumericRange([[-20]])]
        ranges = np.array([rang])
        prob = InferenceRange.piecewise_likelihood_range(pwl, ranges)[0][0]
        self.assertAlmostEqual(prob, 0)


        rang = [NumericRange([[20, 100]])]
        ranges = np.array([rang])
        prob = InferenceRange.piecewise_likelihood_range(pwl, ranges)[0][0]
        self.assertAlmostEqual(prob, 0.493416517396)
        
        rang = [NumericRange([[-20, 20]])]
        ranges = np.array([rang])
        prob = InferenceRange.piecewise_likelihood_range(pwl, ranges)[0][0]
        self.assertAlmostEqual(prob, 0.506583482604)
        
        rang = [NumericRange([[-20, 100]])]
        ranges = np.array([rang])
        prob = InferenceRange.piecewise_likelihood_range(pwl, ranges)[0][0]
        self.assertAlmostEqual(prob, 1)
        
        rang = [NumericRange([[-20, -10]])]
        ranges = np.array([rang])
        prob = InferenceRange.piecewise_likelihood_range(pwl, ranges)[0][0]
        self.assertAlmostEqual(prob, 0)
        
        
    def test_sample_range(self):
        
        np.random.seed(10)
        data = np.random.normal(20, scale=5, size=1000).reshape((1000,1))
        numpy_data = np.array(data, np.float64)
        meta_types = [MetaType.REAL]
        domains = [[np.min(numpy_data[:, 0]), np.max(numpy_data[:, 0])]]
        ds_context = Context(meta_types=meta_types, domains=domains)
        rand_gen = np.random.RandomState(100)
        pwl = create_piecewise_leaf(data, ds_context, scope=[0], prior_weight=None)
        
        rang = [NumericRange([[20]])]
        ranges = np.array(rang)
        samples = SamplingRange.sample_piecewise_node(pwl, 10, rand_gen, ranges)
        self.assertEqual(len(samples), 10)
        self.assertAlmostEqual(np.average(samples), 20)
        
        rang = [NumericRange([[20, 100]])]
        ranges = np.array(rang)
        samples = SamplingRange.sample_piecewise_node(pwl, 10, rand_gen, ranges)
        self.assertTrue(all(samples[samples>20]))
        self.assertTrue(all(samples[samples<100]))
        
        rang = [NumericRange([[10,13],[20, 100]])]
        ranges = np.array(rang)
        samples = SamplingRange.sample_piecewise_node(pwl, 10, rand_gen, ranges)
        self.assertFalse(any(samples[np.where((samples > 13) & (samples < 20))]))
        self.assertFalse(any(samples[samples<10]))



if __name__ == '__main__':
    unittest.main()
