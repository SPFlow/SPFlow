#!/usr/bin/env python3

import unittest
import spflow
from spflow.base.structure.network_type import SPN, BN, UnspecifiedNetworkType


class TestNetworkTypeContextManager(unittest.TestCase):
    def test_network_type_defaul_value(self):
        # Initial value
        self.assertEqual(type(spflow.get_network_type()), UnspecifiedNetworkType)

    def test_network_type_one_context(self):
        # Open first context
        with spflow.set_network_type(SPN()):
            self.assertEqual(type(spflow.get_network_type()), SPN)

    def test_network_type_one_context_reset(self):
        with spflow.set_network_type(SPN()):
            pass

        self.assertEqual(type(spflow.get_network_type()), UnspecifiedNetworkType)

    def test_network_type_sub_context(self):
        # Open first context
        with spflow.set_network_type(SPN()):
            # Open second (sub)context
            with spflow.set_network_type(BN()):
                self.assertEqual(type(spflow.get_network_type()), BN)

    def test_network_type_sub_context_reset(self):
        # Open first context
        with spflow.set_network_type(SPN()):
            pass

            # Open second (sub)context
            with spflow.set_network_type(BN()):
                pass

            self.assertEqual(type(spflow.get_network_type()), SPN)

        self.assertEqual(type(spflow.get_network_type()), UnspecifiedNetworkType)


if __name__ == "__main__":
    unittest.main()
