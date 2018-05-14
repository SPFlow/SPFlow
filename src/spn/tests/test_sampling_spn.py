import unittest

from numpy.random.mtrand import RandomState

from spn.algorithms.Inference import add_node_likelihood, log_likelihood
from spn.algorithms.Posteriors import *
from spn.algorithms.Sampling import sample_induced_trees, sample_spn_weights
from spn.structure.Base import assign_ids
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.Sampling import sample_parametric_node

from scipy.stats import chisquare


def constant_equal_ll(node, data, dtype=np.float64, node_likelihood=None):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    probs[:] = 0.5
    return np.log(probs)


def node_fixed_ll(node, data, dtype=np.float64, node_likelihood=None):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    probs[:] = node.prob
    return np.log(probs)


def leaf(prob, scope=0):
    r = Leaf(scope)
    r.prob = prob
    return r

class TestSampling(unittest.TestCase):
    def check_counts(self, node, count):
        if isinstance(node, Leaf):
            return

        self.assertEqual(len(node.row_ids), count)
        r_id = set(node.row_ids)
        self.assertEqual(len(r_id), len(node.row_ids))

        if isinstance(node, Sum):
            added, r_id_children = 0, set()
            self.assertEqual(np.sum(node.edge_counts), count)
            for i, c in enumerate(node.children):
                self.check_counts(c, node.edge_counts[i])
                added += len(c.row_ids)
                r_id_children.update(c.row_ids)
            self.assertEqual(added, len(r_id_children))
        else:
            for c in node.children:
                self.check_counts(c, count)
                self.assertEqual(r_id, set(c.row_ids))

    def check_induced_tree(self, spn, n=100000):
        rand_gen = RandomState(12345)
        data = rand_gen.rand(n, 3)
        map_rows_cols_to_node_id, lls = sample_induced_trees(spn, data, rand_gen)
        lls_test = np.zeros((data.shape[0], len(get_nodes_by_type(spn))))
        log_likelihood(spn, data, llls_matrix=lls_test)
        self.assertTrue(np.alltrue(np.isclose(lls, lls_test)))
        for s in get_nodes_by_type(spn, Sum):
            expected = np.array(s.weights) * len(s.row_ids)
            observed = s.edge_counts
            stat_test = chisquare(observed, expected)
            print("expected", expected, "observed", observed, "weights", s.weights, "pval", stat_test.pvalue)
            self.assertGreaterEqual(stat_test.pvalue, 0.05)

        # check the map generated
        test_coverage = np.zeros_like(data)
        for r in range(map_rows_cols_to_node_id.shape[0]):
            for c in range(map_rows_cols_to_node_id.shape[1]):
                self.assertEqual(test_coverage[r, c], 0)
                test_coverage[r, c] = 1
        self.assertTrue(np.alltrue(test_coverage == 1))

    def test_induced_trees_correct_parameters(self):
        node_1_2_2 = Leaf(0)
        node_1_2_1 = Leaf(1)
        node_1_1 = Leaf([0, 1])
        node_1_2 = node_1_2_1 * node_1_2_2
        spn = 0.1 * node_1_1 + 0.9 * node_1_2
        node_1_2.id = 0

        rand_gen = RandomState(1234)
        with self.assertRaises(AssertionError):
            sample_induced_trees(spn, rand_gen.rand(10, 3), rand_gen)

        assign_ids(spn)
        node_1_2_2.id += 1

        with self.assertRaises(AssertionError):
            sample_induced_trees(spn, rand_gen.rand(10, 3), rand_gen)

    def test_induced_trees(self):
        add_node_likelihood(Leaf, constant_equal_ll)

        n = 100000

        spn = 0.1 * ((0.1 * (Leaf(0) * Leaf(1)) + 0.9 * (Leaf(0) * Leaf(1))) * Leaf(2)) + 0.9 * (
            (0.8 * (Leaf(0) * Leaf(1)) + 0.2 * (Leaf(0) * Leaf(1))) * Leaf(2))

        self.check_induced_tree(spn, n)
        self.check_counts(spn, n)

        # so far we have tested that the induced trees produce a proper map, does the sampling and the lls are fine
        # we need to check still that the right counts are being passed down accordingly
        # and that the actual computation of the samples and probabilities is correct

        spn = 0.3 * (0.0001 * (Leaf(0) * Leaf(1)) + 0.9999 * (Leaf(0) * Leaf(1))) + 0.7 * (
            0.1 * (0.3 * (Leaf(0) * Leaf(1)) + 0.7 * (Leaf(0) * Leaf(1))) + 0.9 * (
                0.4 * (Leaf(0) * Leaf(1)) + 0.6 * (Leaf(0) * Leaf(1))))

        self.check_induced_tree(spn, n)
        self.check_counts(spn, n)


        l = 0.1 * leaf(0.0001) + 0.9 * leaf(0.99)
        r = 0.1 * leaf(0.5) + 0.9 * leaf(0.1)
        spn = 0.1 * l + 0.9 * r
        add_node_likelihood(Leaf, node_fixed_ll)
        rand_gen = RandomState(12345)
        data = rand_gen.rand(n, 2)
        sample_induced_trees(spn, data, rand_gen)
        self.check_counts(spn, n)

        s = 0.1 * (0.1 * 0.0001 + 0.9 * 0.99) + 0.9 * (0.1 * 0.5 + 0.9 * 0.1)
        expected_freq = np.array([0.1 * (0.1 * 0.0001 + 0.9 * 0.99), 0.9 * (0.1 * 0.5 + 0.9 * 0.1)]) / s
        expected_obs = expected_freq * n
        self.assertGreaterEqual(chisquare(spn.edge_counts, expected_obs).pvalue, 0.05)

        s = 0.1 * 0.0001 + 0.9 * 0.99
        expected_freq = np.array([0.1 * 0.0001, 0.9 * 0.99]) / s
        expected_obs = expected_freq * spn.edge_counts[0]
        self.assertGreaterEqual(chisquare(l.edge_counts, expected_obs).pvalue, 0.05)

        s = 0.1 * 0.5 + 0.9 * 0.1
        expected_freq = np.array([0.1 * 0.5, 0.9 * 0.1]) / s
        expected_obs = expected_freq * spn.edge_counts[1]
        self.assertGreaterEqual(chisquare(r.edge_counts, expected_obs).pvalue, 0.05)

    def test_sample_spn_weights(self):
        n = 100000
        l = 0.1 * leaf(0.0001) + 0.9 * leaf(0.99)
        r = 0.3 * leaf(0.5) + 0.7 * leaf(0.1)
        spn = 0.1 * l + 0.9 * r
        add_node_likelihood(Leaf, node_fixed_ll)
        rand_gen = RandomState(12345)
        data = rand_gen.rand(n, 2)
        sample_induced_trees(spn, data, rand_gen)

        sample_spn_weights(spn, rand_gen, omega_uninf_prior=0)

        remaining_instances = n - spn.edge_counts[0]  # this are the unseen instances for L
        expected_obs = np.array(l.edge_counts)
        expected_obs[0] += 0.1 * remaining_instances
        expected_obs[1] += 0.9 * remaining_instances
        self.assertGreaterEqual(chisquare(np.array(l.weights) * n, expected_obs).pvalue, 0.05)

        remaining_instances = n - spn.edge_counts[1]  # this are the unseen instances for R
        expected_obs = np.array(r.edge_counts)
        expected_obs[0] += 0.3 * remaining_instances
        expected_obs[1] += 0.7 * remaining_instances
        self.assertGreaterEqual(chisquare(np.array(r.weights) * n, expected_obs).pvalue, 0.05)




if __name__ == '__main__':
    unittest.main()
