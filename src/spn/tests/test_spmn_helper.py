import sys


sys.path.append('/home/hari/Desktop/Projects/Thesis_Project/SPFlow_clone/SPFlow/src/')
import unittest
import logging
import numpy as np
from spn.algorithms.SPMNHelper import split_on_decision_node
from spn.algorithms.SPMNHelper import column_slice_data_by_scope
from spn.algorithms.SPMNHelper import get_ds_context
from spn.algorithms.SPMN import SPMNParams
from spn.algorithms.SPMNRework import SPMN

logging.basicConfig(level=logging.DEBUG)


class TestSPMN(unittest.TestCase):

    def setUp(self):
        feature_names = ['X0', 'X1', 'x2', 'D0', 'X3', 'X4', 'X5', 'D1', 'X6', 'U']
        partial_order = [['X0', 'X1', 'x2'], ['D0'], ['X3', 'X4', 'X5'], ['D1'], ['X6', 'X7', 'U']]
        decision_nodes = ['D0', 'D1']
        utility_node = ['U']
        util_to_bin = False

        self.spmn = SPMN(partial_order, decision_nodes, utility_node, feature_names, util_to_bin)

        x012_data = np.arange(30).reshape(10, 3)
        d0_data = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).reshape(10, 1)
        x345_data = np.arange(30, 60).reshape(10, 3)
        d1_data = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).reshape(10, 1)
        x6u_data = np.arange(60, 80).reshape(10, 2)

        self.data = np.concatenate((x012_data, d0_data, x345_data, d1_data, x6u_data), axis=1)

    def test_column_slice_data_by_scope_data10x7_dataScope3456789_sliceScope4678(self):

        data = self.data[:, 3:]
        logging.debug(f'initial data is {data}')
        data_scope = [3, 4, 5, 6, 7, 8, 9]
        slice_scope = [4, 6, 7, 8]
        slice_data = column_slice_data_by_scope(data, data_scope, slice_scope)
        logging.debug(f'sliced data is {slice_data}')
        self.assertEqual((10, 4), slice_data.shape, msg=f'sliced data shape should be (10, 4),'
                                                        f' instead it is {slice_data.shape}')

    def test_split_on_decision_node_decVal01_data10x3(self):

        data = self.data[:, 7:]
        np.random.shuffle(data)
        logging.debug(f'initial data is {data}')

        clusters, dec_vals = split_on_decision_node(data)
        logging.debug(f'clustered data is {clusters}')

        self.assertEqual((5, 2), clusters[0].shape, msg=f'clustered data shape should be (4, 3),'
                                                        f' instead it is {clusters[0].shape}')
        self.assertEqual((5, 2), clusters[1].shape, msg=f'clustered data shape should be (4, 3),'
                                                        f' instead it is {clusters[0].shape}')
        self.assertListEqual([0, 1], dec_vals.tolist(), msg=f'decision values should be [0, 1]'
                                                            f' instead it is {dec_vals.tolist()}')

    def test_get_ds_context(self):

        data = self.data[:, 4:9]
        num_of_cols = data.shape[1]
        logging.debug(f'data {data}')

        scope = [4, 5, 6, 7, 8]
        feature_names = self.spmn.params.feature_names
        util_to_bin = self.spmn.params.util_to_bin
        utility_node = self.spmn.params.utility_node

        params = SPMNParams(utility_node=utility_node, feature_names=feature_names, util_to_bin=util_to_bin)

        ds_context = get_ds_context(data, scope, params)
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        logging.debug(f'meta types {meta_types}')
        logging.debug(f'domains {domains}')

        self.assertEqual(num_of_cols, len(meta_types))
        self.assertEqual(num_of_cols, len(domains))


if __name__ == '__main__':
    unittest.main()
