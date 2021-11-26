import unittest
from spflow.base.memoize import memoize
from spflow.torch.inference import log_likelihood
from spflow.torch.structure.nodes import TorchProductNode, TorchSumNode, TorchGaussian
import sys
import torch


class TestMemoization(unittest.TestCase):
    def test_memoization(self):

        #       [A]         (sum node)
        #       / \
        #      /   \
        #    [B]   [C]      (product nodes)
        #      \   /
        #       \ /
        #       [D]         (sum nodes)
        #       / \
        #      /   \
        #    [E]   [F]      (product nodes)
        #     |\   /|
        #     | \ / |
        #     |  X  |
        #     | / \ |
        #     |/   \|
        #    [G]   [H]      (leaf nodes)

        H = TorchGaussian([0], 0.0, 1.0)
        G = TorchGaussian([1], 0.0, 1.0)

        F = TorchProductNode([G, H], [0])
        E = TorchProductNode([G, H], [0])

        D = TorchSumNode([E, F], [0])

        C = TorchProductNode([D], [0])
        B = TorchProductNode([D], [0])

        A = TorchSumNode([B, C], [0])

        data = torch.randn(3, 2)

        hist = []

        def tracefunc(frame, event, arg, hist=hist):
            if event == "call" and frame.f_code.co_name in ["memoized_f", "log_likelihood"]:
                hist.append(frame.f_code.co_name)
            return tracefunc

        sys.settrace(tracefunc)

        log_likelihood(A, data)

        sys.settrace(None)

        self.assertTrue(
            hist
            == [
                "memoized_f",
                "log_likelihood",  # node A not cached: A
                "memoized_f",
                "log_likelihood",  # node B not cached: A -> B
                "memoized_f",
                "log_likelihood",  # node D not cached: A -> B -> D
                "memoized_f",
                "log_likelihood",  # node E not cached: A -> B -> D -> E
                "memoized_f",
                "log_likelihood",  # node G not cached: A -> B -> D -> E -> G
                "memoized_f",
                "log_likelihood",  # node H not cached: A -> B -> D -> E -> H
                "memoized_f",
                "log_likelihood",  # node F not cached: A -> B -> D -> F
                "memoized_f",  # node G cached:     A -> B -> D -> F -> G
                "memoized_f",  # node H cached:     A -> B -> D -> F -> H
                "memoized_f",
                "log_likelihood",  # node C not cached: A -> C
                "memoized_f",  # node D cached:     A -> C -> D
            ]
        )


if __name__ == "__main__":
    unittest.main()
