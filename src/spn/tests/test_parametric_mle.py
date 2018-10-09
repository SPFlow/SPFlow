import json
import unittest

from spn.io.Text import to_JSON
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import *


class TestParametric(unittest.TestCase):
    def setUp(self):
        self.tested = set()

        # print(to_JSON(CategoricalDictionary(p={0: 0.5, 5: 0.5}, scope=0)))
        # 0/0

    def assert_correct(self, expected, data):
        self.tested.add(type(expected))

        domains = [[np.min(data), np.max(data)]]

        mle = create_parametric_leaf(data, ds_context=Context(parametric_types=[type(expected)], domains=domains),
                                     scope=[0])

        a = to_JSON(mle)
        b = to_JSON(expected)
        self.assertEqual(a, b)

        return mle

    def test_Parametric_inference(self):
        data = np.array([0, 0, 5, 5]).reshape(-1, 1)
        self.assert_correct(CategoricalDictionary(p={0: 0.5, 5: 0.5}, scope=0), data)

        data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        self.assert_correct(Gaussian(mean=np.mean(data), stdev=np.std(data), scope=0), data)

        node = self.assert_correct(Gamma(alpha=3.701643810008814, beta=1.233881270002938, scope=0), data)
        self.assertEqual(node.alpha / node.beta, np.mean(data))

        self.assert_correct(LogNormal(mean=np.log(data).mean(), stdev=np.log(data).std(), scope=0), data)

        self.assert_correct(Poisson(mean=np.mean(data), scope=0), data)

        self.assert_correct(Exponential(l=np.mean(data), scope=0), data)

        self.assert_correct(Geometric(p=1.0 / np.mean(data), scope=0), data)

        data = np.array([0, 0, 1, 3, 5, 6, 6, 6, 6, 6]).reshape(-1, 1)
        self.assert_correct(Categorical(p=[2 / 10, 1 / 10, 0 / 10, 1 / 10, 0 / 10, 1 / 10, 5 / 10], scope=0), data)

        data = np.array([0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
        self.assert_correct(Bernoulli(p=0.75, scope=0), data)

        for child in Parametric.__subclasses__():
            if child not in self.tested:
                print("not tested", child)


if __name__ == '__main__':
    unittest.main()
